import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 8192, eps: float = 1e-6):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _rms(x: Tensor, eps: float) -> Tensor:
        return (x.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()

    def forward(self, x: Tensor, position_ids: Optional[Tensor] = None, valid_mask: Optional[Tensor] = None) -> Tensor:
        B, S, D = x.shape
        if position_ids is None:
            position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)

        pos = self.pe.index_select(0, position_ids.reshape(-1)).view(B, S, D).to(dtype=x.dtype)

        x_rms = self._rms(x, self.eps).detach()
        pos_rms = self._rms(pos, self.eps)
        pos = pos * (x_rms / (pos_rms + self.eps))

        out = x + pos

        if valid_mask is not None:
            out = out * valid_mask.to(out.dtype).unsqueeze(-1)

        return self.dropout(out)

class SRPFN(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        emsize: int,
        interaction_encoder: Callable[[int, int], nn.Module],
        nhead: int,
        nhid: int,
        nlayers: int,
        max_history_len: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        input_normalization: bool = False,
        interaction_tuple: int = 5,
    ):
        super().__init__()
        self.model_type = "TransformerRegression_CrossAttn"
        self.ninp = emsize
        self.max_history_len = max_history_len

        self.attn_mask_mode = "bool"
        
        self.interaction_encoder = interaction_encoder(
            input_dim=feature_dim * interaction_tuple,
            output_dim=emsize
        )

        self.evidence_attn = nn.MultiheadAttention(
            embed_dim=self.ninp,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.evidence_ln = nn.LayerNorm(self.ninp)

        self.delta_gate = nn.Sequential(
            nn.LayerNorm(self.ninp),
            nn.Linear(self.ninp, self.ninp),
            nn.GELU(),
            nn.Linear(self.ninp, 1),
        )
        nn.init.zeros_(self.delta_gate[-1].weight)
        nn.init.constant_(self.delta_gate[-1].bias, 0.0)

        self.posterior_head = nn.Sequential(
            nn.LayerNorm(self.ninp),
            nn.Linear(self.ninp, self.ninp, bias=False),
        )

        self.pos_encoder = PositionalEncoding(self.ninp, dropout)
        self.input_ln = nn.LayerNorm(self.ninp) if input_normalization else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=nlayers,
            norm=nn.LayerNorm(self.ninp),
        )

        self.next_item_head = nn.Sequential(
            nn.LayerNorm(self.ninp),
            nn.Linear(self.ninp, self.ninp, bias=False),
        )
        nn.init.xavier_uniform_(self.next_item_head[1].weight, gain=0.1)

        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def _make_causal_mask_2d(self, S: int, device: torch.device, dtype: torch.dtype):
        m_bool = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)
        if self.attn_mask_mode == "bool":
            return m_bool
        m = torch.zeros(S, S, device=device, dtype=dtype)
        m.masked_fill_(m_bool, float("-inf"))
        return m

    def encode_query_sequence(
        self,
        interaction_vectors: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        B, S, _ = interaction_vectors.shape
        device = interaction_vectors.device

        valid_mask = attention_mask.to(torch.bool)
        key_padding_mask = ~valid_mask

        final_input = torch.zeros(B, S, self.ninp, device=device, dtype=torch.float32)

        if valid_mask.any():
            x = interaction_vectors[valid_mask]
            enc = self.interaction_encoder(x)
            final_input[valid_mask] = enc.to(final_input.dtype)

        final_input = self.pos_encoder(final_input, position_ids, valid_mask=valid_mask)
        final_input = final_input * valid_mask.unsqueeze(-1).to(final_input.dtype)

        if self.input_ln is not None:
            final_input = self.input_ln(final_input)

        causal_2d = self._make_causal_mask_2d(S, device=final_input.device, dtype=final_input.dtype)

        out = self.transformer_encoder(
            final_input,
            mask=causal_2d,
            src_key_padding_mask=key_padding_mask
        )
        return out

    def build_evidence_memory(
        self,
        support_interaction_vecs: Tensor,
        support_valid_mask: Optional[Tensor] = None,
        detach_mem: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        B, K, _ = support_interaction_vecs.shape
        device = support_interaction_vecs.device

        if support_valid_mask is None:
            valid = torch.ones((B, K), dtype=torch.bool, device=device)
        else:
            valid = support_valid_mask.to(torch.bool)

        key_padding = ~valid

        mem = torch.zeros((B, K, self.ninp), device=device, dtype=torch.float32)
        if valid.any():
            flat_x = support_interaction_vecs.view(B * K, -1)
            flat_v = valid.view(B * K)
            enc = self.interaction_encoder(flat_x[flat_v])
            if detach_mem:
                enc = enc.detach()
            mem.view(B * K, -1)[flat_v] = enc.to(mem.dtype)

        return mem, key_padding

    def compute_delta_via_cross_attn(
        self,
        q_hidden: Tensor,
        answer_positions: Tensor,
        mem: Tensor,
        key_padding: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        B, S, D = q_hidden.shape
        device = q_hidden.device

        if mem.size(1) == 0:
            delta = torch.zeros(B, self.ninp, device=device, dtype=q_hidden.dtype)
            msg_vec = torch.zeros(B, D, device=device, dtype=q_hidden.dtype)
            return delta, None, msg_vec

        q = q_hidden[torch.arange(B, device=device), answer_positions].unsqueeze(1)

        all_pad = key_padding.all(dim=1)
        if all_pad.any():
            key_padding = key_padding.clone()
            mem = mem.clone()
            key_padding[all_pad, 0] = False
            mem[all_pad, 0].zero_()

        msg, attn = self.evidence_attn(
            query=self.evidence_ln(q),
            key=mem,
            value=mem,
            key_padding_mask=key_padding,
            need_weights=True,
        )

        if all_pad.any():
            msg = msg.clone()
            attn = attn.clone()
            msg[all_pad] = 0.0
            attn[all_pad] = 0.0

        msg_vec = msg.squeeze(1)
        delta = self.posterior_head(msg_vec)

        if all_pad.any():
            delta = delta.clone()
            delta[all_pad] = 0.0

        return delta, attn, msg_vec

    def predict_next_vector(self, query_transformer_output: Tensor, answer_positions: Tensor) -> Tensor:
        B = query_transformer_output.size(0)
        device = query_transformer_output.device
        h_ans = query_transformer_output[torch.arange(B, device=device), answer_positions]
        pred = self.next_item_head(h_ans)
        pred = F.normalize(pred, dim=-1, eps=1e-8)
        return pred

    def score_candidates(
        self,
        q_prior: Tensor,
        delta: Tensor,
        g: Tensor,
        candidate_interaction_vecs: Tensor,
        candidate_counts: Optional[Tensor],
        return_parts: bool = False,
        delta_scale: float = 1.0,
    ):
        B, C, _ = candidate_interaction_vecs.shape
        scale = self.logit_scale.exp().clamp(1.0, 100.0)

        cand_flat = candidate_interaction_vecs.reshape(B * C, -1)
        cand_encoded_flat = self.interaction_encoder(cand_flat)
        cand = F.normalize(cand_encoded_flat.view(B, C, -1).to(q_prior.dtype), dim=-1)

        q = F.normalize(q_prior, dim=-1, eps=1e-8).unsqueeze(1)
        d = (g.to(q_prior.dtype) * delta.to(q_prior.dtype)).unsqueeze(1)

        logits_prior = torch.bmm(q, cand.transpose(1, 2)).squeeze(1) * scale
        logits_delta = torch.bmm(d, cand.transpose(1, 2)).squeeze(1) * float(delta_scale)

        logits = logits_prior + logits_delta

        if candidate_counts is not None:
            mask = torch.arange(C, device=logits.device).unsqueeze(0) >= candidate_counts.unsqueeze(1)
            logits = logits.masked_fill(mask, float("-inf"))

        if return_parts:
            return logits, logits_prior, logits_delta
        return logits

    def score_all_items(
        self,
        q_prior: Tensor,
        delta: Tensor,
        g: Tensor,
        query_user_emb: Tensor,
        query_prev_item_emb: Tensor,
        query_prev_trans_emb: Tensor,
        all_item_cf: Tensor,
        all_trans_c: Tensor,
        num_items: int,
        return_parts: bool = False,
        delta_scale: float = 1.0,
        chunk_size: int = 1024,
        temperature: float = 0.07,
    ):
        B = q_prior.size(0)
        F_dim = query_user_emb.size(1)
        M = int(num_items)
        device = q_prior.device
        scale = self.logit_scale.exp().clamp(0.1, 100.0)
        temperature = float(temperature)
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        q = F.normalize(q_prior, dim=-1, eps=1e-8).unsqueeze(1)
        d = (g.to(q_prior.dtype) * delta.to(q_prior.dtype)).unsqueeze(1)

        logits_all = []
        logits_prior_all = []
        logits_delta_all = []

        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            c_chunk = end - start

            item_cf_chunk = all_item_cf[start:end]
            trans_c_chunk = all_trans_c[start:end]

            chunk_vecs = torch.empty(B, c_chunk, 5 * F_dim, device=device, dtype=q_prior.dtype)
            chunk_vecs[..., 0*F_dim:1*F_dim] = query_user_emb.unsqueeze(1)
            chunk_vecs[..., 1*F_dim:2*F_dim] = query_prev_item_emb.unsqueeze(1)
            chunk_vecs[..., 2*F_dim:3*F_dim] = item_cf_chunk.unsqueeze(0)
            chunk_vecs[..., 3*F_dim:4*F_dim] = query_prev_trans_emb.unsqueeze(1)
            chunk_vecs[..., 4*F_dim:5*F_dim] = trans_c_chunk.unsqueeze(0)

            flat = chunk_vecs.reshape(B * c_chunk, -1)
            encoded = self.interaction_encoder(flat)
            cand = F.normalize(encoded.view(B, c_chunk, -1).to(q_prior.dtype), dim=-1)

            logits_prior_chunk = torch.bmm(q, cand.transpose(1, 2)).squeeze(1) * scale
            logits_delta_chunk = torch.bmm(d, cand.transpose(1, 2)).squeeze(1) * float(delta_scale)
            logits_prior_chunk = logits_prior_chunk / temperature
            logits_delta_chunk = logits_delta_chunk / temperature
            logits_chunk = logits_prior_chunk + logits_delta_chunk
            logits_all.append(logits_chunk)

            if return_parts:
                logits_prior_all.append(logits_prior_chunk)
                logits_delta_all.append(logits_delta_chunk)

            del chunk_vecs, flat, encoded, cand

        logits = torch.cat(logits_all, dim=1)
        logits_prior = torch.cat(logits_prior_all, dim=1) if return_parts else None
        logits_delta = torch.cat(logits_delta_all, dim=1) if return_parts else None

        if return_parts:
            return logits, logits_prior, logits_delta
        return logits

    def forward(
        self,
        interaction_vectors: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        answer_positions: Tensor,
        support_interaction_vecs: Tensor,
        support_valid_mask: Optional[Tensor] = None,
        candidate_interaction_vecs: Optional[Tensor] = None,
        candidate_counts: Optional[Tensor] = None,
        query_user_emb: Optional[Tensor] = None,
        query_prev_item_emb: Optional[Tensor] = None,
        query_prev_trans_emb: Optional[Tensor] = None,
        all_item_cf: Optional[Tensor] = None,
        all_trans_c: Optional[Tensor] = None,
        num_items: Optional[int] = None,
        item_chunk_size: int = 1024,
        full_item_temperature: float = 0.07,
        return_logits: bool = False,
        return_pred: bool = True,
        return_parts: bool = False,
        detach_mem: bool = True,
        delta_scale: float = 1.0,
    ):
        query_out = self.encode_query_sequence(
            interaction_vectors=interaction_vectors,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        q_prior = self.predict_next_vector(query_out, answer_positions)

        mem, key_padding = self.build_evidence_memory(
            support_interaction_vecs=support_interaction_vecs,
            support_valid_mask=support_valid_mask,
            detach_mem=detach_mem,
        )

        delta, attn_evi, msg_vec = self.compute_delta_via_cross_attn(
            q_hidden=query_out,
            answer_positions=answer_positions,
            mem=mem,
            key_padding=key_padding,
        )

        logit = self.delta_gate(q_prior)
        g = torch.sigmoid(logit)

        no_support = key_padding.all(dim=1)
        g = torch.where(no_support.unsqueeze(1), torch.zeros_like(g), g).clamp(0.0, 1.0)

        q_post = F.normalize(q_prior + g * delta, dim=-1, eps=1e-8)

        out = {}
        if return_pred:
            out["pred_vec"] = q_post

        out.update({
            "g": g,
            "delta": delta,
            "msg_vec": msg_vec,
            "attn_evi": attn_evi,
            "key_padding": key_padding,
            "q_prior": q_prior,
        })

        if return_logits:
            if all_item_cf is not None and num_items is not None:
                score_out = self.score_all_items(
                    q_prior=q_prior,
                    delta=delta,
                    g=g,
                    query_user_emb=query_user_emb,
                    query_prev_item_emb=query_prev_item_emb,
                    query_prev_trans_emb=query_prev_trans_emb,
                    all_item_cf=all_item_cf,
                    all_trans_c=all_trans_c,
                    num_items=num_items,
                    return_parts=return_parts,
                    delta_scale=delta_scale,
                    chunk_size=item_chunk_size,
                    temperature=full_item_temperature,
                )
                if return_parts:
                    logits, logits_prior, logits_delta = score_out
                    out["logits"] = logits
                    out["logits_prior"] = logits_prior
                    out["logits_delta"] = logits_delta
                else:
                    out["logits"] = score_out
            elif candidate_interaction_vecs is not None:
                logits, logits_prior, logits_delta = self.score_candidates(
                    q_prior=q_prior,
                    delta=delta,
                    g=g,
                    candidate_interaction_vecs=candidate_interaction_vecs,
                    candidate_counts=candidate_counts,
                    return_parts=True,
                    delta_scale=delta_scale,
                )
                out["logits"] = logits
                if return_parts:
                    out["logits_prior"] = logits_prior
                    out["logits_delta"] = logits_delta

        if len(out) == 1 and "pred_vec" in out:
            return out["pred_vec"]
        return out
