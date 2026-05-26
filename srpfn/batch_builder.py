from typing import Callable, Dict, List, Optional

import torch
from tqdm import tqdm

PAD_ID = -1

class BatchBuilder:
    def build_batches_for_users(
        self,
        selected_uids: List[int],
        prior_state: Dict[str, object],
        user_sequences: Dict[int, List[int]],
        num_users: int,
        batch_size: int,
        support_set_size: int,
        predefined_candidates: Optional[Dict[int, List[int]]] = None,
        mode: str = "cand",
        seed_callback: Optional[Callable[[int], None]] = None,
        progress_desc: str = "Generating batches",
        progress_file=None,
    ) -> List[dict]:
        all_batches = []
        if not selected_uids:
            return all_batches

        sequences_obs = prior_state["sequences_obs"]
        sequences_full = prior_state["sequences_full"]
        top_idx = prior_state["top_idx"]
        top_p = prior_state["top_p"]
        embeddings = prior_state["embeddings"]
        transition_item_count = prior_state["transition_item_count"]
        device = self.device

        num_batches = (len(selected_uids) + batch_size - 1) // batch_size
        batch_iter = range(0, len(selected_uids), batch_size)
        for batch_idx, i in enumerate(
            tqdm(batch_iter, desc=progress_desc, total=num_batches, unit="batch", file=progress_file)
        ):
            batch_uids = selected_uids[i:i + batch_size]
            if not batch_uids:
                continue
            if max(batch_uids) >= num_users:
                continue

            q_batch = torch.tensor(batch_uids, dtype=torch.long, device=device)

            try:
                if seed_callback is not None:
                    seed_callback(batch_idx)

                final_batch = self._generate_and_create_batch(
                    query_uids_batch=q_batch,
                    sequences_obs=sequences_obs,
                    sequences_full=sequences_full,
                    top_idx=top_idx,
                    top_p=top_p,
                    M=transition_item_count,
                    embeddings=embeddings,
                    support_set_size=support_set_size,
                )

                if final_batch is None:
                    continue

                if mode == "full":
                    all_batches.append(final_batch)
                    continue

                if predefined_candidates is not None:
                    B_actual = final_batch["candidate_items"].size(0)

                    obs_offsets = sequences_obs["offsets"]
                    full_offsets = sequences_full["offsets"]
                    obs_starts = obs_offsets[q_batch]
                    obs_ends = obs_offsets[q_batch + 1]
                    obs_lens = obs_ends - obs_starts
                    full_starts = full_offsets[q_batch]
                    full_ends = full_offsets[q_batch + 1]
                    full_lens = full_ends - full_starts
                    valid_mask = (full_lens >= 2) & (obs_lens >= 1)
                    valid_uids_tensor = q_batch[valid_mask]

                    if valid_uids_tensor.numel() != B_actual:
                        B_actual = min(B_actual, valid_uids_tensor.numel())
                        valid_uids_tensor = valid_uids_tensor[:B_actual]

                    candidate_rows = []
                    for uid in valid_uids_tensor.tolist():
                        pos_item = user_sequences[uid][-1]
                        neg_items = predefined_candidates[uid]
                        candidate_rows.append([pos_item] + neg_items)

                    new_candidates = torch.tensor(candidate_rows, dtype=torch.long, device=device)
                    C_new = new_candidates.size(1)

                    final_batch["candidate_items"] = new_candidates
                    final_batch["candidates"] = new_candidates
                    final_batch["labels"] = torch.zeros(B_actual, dtype=torch.long, device=device)
                    final_batch["candidate_counts"] = torch.full((B_actual,), C_new, device=device, dtype=torch.long)

                    obs_ends_valid = obs_offsets[valid_uids_tensor + 1]
                    cand_prev_states = sequences_obs["flat"][obs_ends_valid - 1]
                    c_u = embeddings["user_cf"][valid_uids_tensor].unsqueeze(1).expand(-1, C_new, -1)
                    c_i_prev = embeddings["item_cf"][cand_prev_states].unsqueeze(1).expand(-1, C_new, -1)
                    c_prev = embeddings["trans_r"][cand_prev_states].unsqueeze(1).expand(-1, C_new, -1)
                    c_i_curr = embeddings["item_cf"][new_candidates]
                    c_next = embeddings["trans_c"][new_candidates]

                    final_batch["candidate_interaction_vecs"] = torch.cat(
                        [c_u, c_i_prev, c_i_curr, c_prev, c_next], dim=-1
                    )
                else:
                    final_batch["candidates"] = final_batch["candidate_items"]

                batch_cpu = {
                    k: v.to("cpu") if isinstance(v, torch.Tensor) else v
                    for k, v in final_batch.items()
                }
                all_batches.append(batch_cpu)

            except Exception:
                continue

        return all_batches

    def _vectorized_candidate_sampling(self, q_answers, num_negatives, M, device):
        B = q_answers.size(0)
        
        neg_candidates = torch.randint(0, M, (B, num_negatives), device=device)
        
        mask_collision = (neg_candidates == q_answers.unsqueeze(1))
        
        if mask_collision.any():
            num_collisions = mask_collision.sum().item()
            replacements = torch.randint(0, M, (num_collisions,), device=device)
            neg_candidates[mask_collision] = replacements
            
        candidates = torch.cat([q_answers.unsqueeze(1), neg_candidates], dim=1)
        
        rand_vals = torch.rand((B, 1 + num_negatives), device=device)
        perm_indices = torch.argsort(rand_vals, dim=1)
        
        shuffled_candidates = torch.gather(candidates, 1, perm_indices)
        
        labels = (shuffled_candidates == q_answers.unsqueeze(1)).float().argmax(dim=1)
        
        return shuffled_candidates, labels

    def _sample_support_set(
        self,
        history_padded: torch.Tensor,
        history_lengths: torch.Tensor,
        top_idx: torch.Tensor,
        top_p: torch.Tensor,
        support_set_size: int,
    ):
        device = history_padded.device
        B = history_padded.shape[0]
        M = top_idx.shape[0]
        pad_id = PAD_ID

        safe_lengths = history_lengths.clamp(min=1)
        last_indices = (safe_lengths - 1).clamp(max=history_padded.size(1) - 1)
        last_items = torch.gather(history_padded, 1, last_indices.unsqueeze(1)).squeeze(1)
        last_items = torch.where(last_items == pad_id, torch.zeros_like(last_items), last_items)

        neigh_idx = top_idx[last_items].long() 
        neigh_p = top_p[last_items].clone()

        valid_mask = (neigh_idx >= 0)
        neigh_p[~valid_mask] = 0.0
        
        row_sum = neigh_p.sum(dim=1, keepdim=True)
        is_isolated = (row_sum.squeeze(-1) == 0)
        neigh_p[is_isolated] = 1.0 
        neigh_p = neigh_p / neigh_p.sum(dim=1, keepdim=True)

        are_deterministic = torch.are_deterministic_algorithms_enabled()
        if are_deterministic:
            torch.use_deterministic_algorithms(False)
            
        try:
            sampled_indices = torch.multinomial(neigh_p, support_set_size, replacement=True)
        finally:
            if are_deterministic:
                torch.use_deterministic_algorithms(True)

        support_items = torch.gather(neigh_idx, 1, sampled_indices)

        if is_isolated.any():
            support_items[is_isolated] = last_items[is_isolated].unsqueeze(1).expand(-1, support_set_size)

        invalid_sampled = (support_items == pad_id) | (support_items < 0)
        if invalid_sampled.any():
            fallback = last_items.unsqueeze(1).expand(-1, support_set_size)
            support_items = torch.where(invalid_sampled, fallback, support_items)

        return support_items

    def _generate_and_create_batch(
        self,
        query_uids_batch,
        sequences_obs,
        sequences_full,
        top_idx,
        top_p,
        M,
        embeddings,
        support_set_size,
    ):
        device = self.device
        
        obs_flat = sequences_obs["flat"]
        obs_offsets = sequences_obs["offsets"]
        full_flat = sequences_full["flat"]
        full_offsets = sequences_full["offsets"]

        obs_starts = obs_offsets[query_uids_batch]
        obs_ends   = obs_offsets[query_uids_batch + 1]
        obs_lens   = obs_ends - obs_starts
        
        full_starts = full_offsets[query_uids_batch]
        full_ends   = full_offsets[query_uids_batch + 1]
        full_lens   = full_ends - full_starts

        valid_mask = (full_lens >= 2) & (obs_lens >= 1)
        
        if not valid_mask.any():
            return None

        valid_uids = query_uids_batch[valid_mask]
        obs_starts = obs_starts[valid_mask]
        obs_ends   = obs_ends[valid_mask]
        obs_lens   = obs_lens[valid_mask]
        full_ends  = full_ends[valid_mask]

        B = valid_uids.size(0)

        eff_lens = obs_lens.clone()
        if self.max_history_len > 0:
            eff_lens = torch.clamp(eff_lens, max=self.max_history_len)
        
        L = eff_lens.max().item()
        
        actual_starts = obs_ends - eff_lens
        
        base_grid = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        gather_idx = actual_starts.unsqueeze(1) + base_grid
        
        seq_mask = base_grid < eff_lens.unsqueeze(1)
        
        safe_gather_idx = gather_idx * seq_mask.long()
        
        padded_i_curr = obs_flat[safe_gather_idx]
        pad_id = PAD_ID
        padded_i_curr = torch.where(seq_mask, padded_i_curr, torch.full_like(padded_i_curr, pad_id))
        
        padded_i_prev = torch.full_like(padded_i_curr, pad_id)
        padded_i_prev[:, 1:] = padded_i_curr[:, :-1]
        
        padded_u_ids = valid_uids.unsqueeze(1).expand(B, L)
        
        attention_mask = seq_mask
        answer_positions = eff_lens - 1
        padded_pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        
        labels = full_flat[full_ends - 1]
        answer_positions = eff_lens - 1
        last_items = padded_i_curr[torch.arange(B, device=device), answer_positions]
        
        cand_prev_states = obs_flat[obs_ends - 1]

        support_interaction_vecs = None
        support_valid_mask = None

        F = self.feature_dim
        
        support_interaction_vecs = torch.zeros((B, 0, 5 * F), device=device, dtype=torch.float32)
        
        if support_set_size > 0:
            actual_K = support_set_size
            try:
                support_targets = self._sample_support_set(
                    history_padded=padded_i_curr,
                    history_lengths=eff_lens,
                    top_idx=top_idx,
                    top_p=top_p,
                    support_set_size=support_set_size
                )
                support_sources = last_items.unsqueeze(1).expand(-1, support_set_size)

                F = self.feature_dim
                support_interaction_vecs = torch.empty((B, actual_K, 5 * F), device=device, dtype=torch.float32)

                u_vec = embeddings['user_cf'][valid_uids].unsqueeze(1).expand(-1, actual_K, -1)
                
                safe_sources = torch.where(support_sources == pad_id, torch.zeros_like(support_sources), support_sources).long()
                i_prev_vec = embeddings['item_cf'][safe_sources]

                safe_targets = torch.where(support_targets == pad_id, torch.zeros_like(support_targets), support_targets).long()
                i_curr_vec = embeddings['item_cf'][safe_targets]

                p_context_vec = embeddings['trans_r'][safe_sources]
                
                c_context_vec = embeddings['trans_c'][safe_targets]

                support_interaction_vecs[..., 0*F:1*F] = u_vec
                support_interaction_vecs[..., 1*F:2*F] = i_prev_vec
                support_interaction_vecs[..., 2*F:3*F] = i_curr_vec
                support_interaction_vecs[..., 3*F:4*F] = p_context_vec
                support_interaction_vecs[..., 4*F:5*F] = c_context_vec

                support_valid_mask = (support_targets != pad_id) & (support_targets >= 0) & (support_targets < M)
                support_interaction_vecs = support_interaction_vecs * support_valid_mask.unsqueeze(-1).float()

            except Exception:
                F5 = 5 * self.feature_dim
                support_interaction_vecs = torch.zeros((B, actual_K, F5), device=device)
                support_valid_mask = torch.zeros((B, actual_K), dtype=torch.bool, device=device)

        safe_padded_i_curr = torch.where(padded_i_curr == pad_id, torch.zeros_like(padded_i_curr), padded_i_curr)
        safe_padded_i_prev = torch.where(padded_i_prev == pad_id, torch.zeros_like(padded_i_prev), padded_i_prev)

        F = self.feature_dim
        interaction_vectors = torch.empty((B, L, 5 * F), device=device, dtype=torch.float32)

        emb_u = embeddings['user_cf'][padded_u_ids]
        emb_i_prev = embeddings['item_cf'][safe_padded_i_prev]
        emb_i_curr = embeddings['item_cf'][safe_padded_i_curr]
        emb_trans_r = embeddings['trans_r'][safe_padded_i_prev]
        emb_trans_c = embeddings['trans_c'][safe_padded_i_curr]

        interaction_vectors[..., 0*F:1*F] = emb_u
        interaction_vectors[..., 1*F:2*F] = emb_i_prev
        interaction_vectors[..., 2*F:3*F] = emb_i_curr
        interaction_vectors[..., 3*F:4*F] = emb_trans_r
        interaction_vectors[..., 4*F:5*F] = emb_trans_c

        interaction_vectors = interaction_vectors * attention_mask.unsqueeze(-1).float()

        if self.scoring_mode == "full_item":
            num_items_total = int(embeddings['item_cf'].size(0))
            query_user_emb = embeddings['user_cf'][valid_uids]
            query_prev_item_emb = embeddings['item_cf'][cand_prev_states]
            query_prev_trans_emb = embeddings['trans_r'][cand_prev_states]
            all_item_cf = embeddings['item_cf']
            all_trans_c = embeddings['trans_c']

            return {
                "interaction_vectors": interaction_vectors,
                "position_ids": padded_pos_ids,
                "attention_mask": attention_mask,
                "answer_positions": answer_positions,
                "support_interaction_vecs": support_interaction_vecs,
                "support_valid_mask": support_valid_mask,
                "labels": labels,
                "query_user_emb": query_user_emb,
                "query_prev_item_emb": query_prev_item_emb,
                "query_prev_trans_emb": query_prev_trans_emb,
                "all_item_cf": all_item_cf,
                "all_trans_c": all_trans_c,
                "num_items": num_items_total,
            }

        cands, sampled_labels = self._vectorized_candidate_sampling(
            labels, self.num_negatives, M, device
        )
        C = cands.size(1)
        
        candidate_interaction_vecs = torch.empty((B, C, 5 * F), device=device, dtype=torch.float32)

        c_u = embeddings['user_cf'][valid_uids].unsqueeze(1).expand(-1, C, -1)
        c_i_prev = embeddings['item_cf'][cand_prev_states].unsqueeze(1).expand(-1, C, -1)
        c_i_curr = embeddings['item_cf'][cands]
        c_prev = embeddings['trans_r'][cand_prev_states].unsqueeze(1).expand(-1, C, -1)
        c_next = embeddings['trans_c'][cands]

        candidate_interaction_vecs[..., 0*F:1*F] = c_u
        candidate_interaction_vecs[..., 1*F:2*F] = c_i_prev
        candidate_interaction_vecs[..., 2*F:3*F] = c_i_curr
        candidate_interaction_vecs[..., 3*F:4*F] = c_prev
        candidate_interaction_vecs[..., 4*F:5*F] = c_next

        candidate_counts = torch.full((B,), C, device=device, dtype=torch.long)
        
        return {
            "interaction_vectors": interaction_vectors,
            "position_ids": padded_pos_ids,
            "attention_mask": attention_mask,
            "answer_positions": answer_positions,
            "support_interaction_vecs": support_interaction_vecs,
            "support_valid_mask": support_valid_mask,
            "labels": sampled_labels,
            "candidate_counts": candidate_counts,
            "candidate_items": cands,
            "candidate_interaction_vecs": candidate_interaction_vecs,
        }
