from typing import Dict, List, Optional

import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np
import torch
from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix

class FeatureBuilder:
    def build_prior_state_from_sequences(
        self,
        user_sequences: Dict[int, List[int]],
        num_users: int,
        num_items: int,
        feature_dim: int,
        seed: int = 0,
        full_item: bool = False,
        topk: int = 512,
    ) -> Dict[str, object]:
        histories = {
            uid: (user_sequences[uid][:-1] if len(user_sequences[uid]) > 1 else [])
            for uid in range(num_users)
        }

        user_item_matrix = self._create_matrix_from_sequences(
            histories, num_users=num_users, num_items=num_items
        )
        coo = user_item_matrix.coalesce()

        cupy_dev_str = f"cuda:{self.cupy_device_id}"
        r_idx_cuda = coo.indices()[0].to(device=cupy_dev_str, dtype=torch.int32)
        c_idx_cuda = coo.indices()[1].to(device=cupy_dev_str, dtype=torch.int32)
        vals_cuda = coo.values().to(device=cupy_dev_str, dtype=torch.float32)

        rows_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(r_idx_cuda))
        cols_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(c_idx_cuda))
        vals_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(vals_cuda))

        interaction_matrix = cupy_csr_matrix((vals_cp, (rows_cp, cols_cp)), shape=coo.shape)
        del user_item_matrix, coo, r_idx_cuda, c_idx_cuda, vals_cuda

        interaction_matrix = self._apply_ppmi_transform(interaction_matrix)
        user_emb, item_emb = self._compute_csr_svd_embeddings_gpu(
            interaction_matrix, feature_dim, omega_seed=int(seed) + 401
        )

        transition_matrix = self._create_transition_matrix_gpu(histories, num_items)
        transition_matrix = self._apply_ppmi_transform(transition_matrix)
        trans_r, trans_c = self._compute_csr_svd_embeddings_gpu(
            transition_matrix, feature_dim, omega_seed=int(seed) + 301
        )

        sequences_full = self._prepare_sequences_on_device(user_sequences)
        sequences_obs = self._prepare_sequences_on_device(histories)

        src_u, dst_u, cnt_u, transition_item_count = self._build_global_transition_counter(
            sequences_obs=sequences_obs,
            num_items=num_items if full_item else None,
        )
        top_idx, top_p = self._build_topk_from_global_counter(
            src_u, dst_u, cnt_u, M=transition_item_count, K=topk
        )

        return {
            "histories": histories,
            "sequences_full": sequences_full,
            "sequences_obs": sequences_obs,
            "top_idx": top_idx,
            "top_p": top_p,
            "transition_item_count": transition_item_count,
            "embeddings": {
                "user_cf": user_emb,
                "item_cf": item_emb,
                "trans_r": trans_r,
                "trans_c": trans_c,
            },
        }

    def _build_global_transition_counter(self, sequences_obs, num_items: Optional[int] = None):
        flat = sequences_obs["flat"]
        offsets = sequences_obs["offsets"]
        device = flat.device

        if flat.numel() == 0:
            M = int(num_items) if num_items is not None else 0
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty, M

        observed_max = int(flat.max().item())
        M = int(num_items) if num_items is not None else (observed_max + 1)
        if num_items is not None and observed_max >= M:
            raise RuntimeError(
                f"Observed item ids exceed num_items: max={observed_max}, num_items={M}"
            )

        end_pos = (offsets[1:-1] - 1).clamp(min=0)
        mask = torch.ones(flat.numel() - 1, device=device, dtype=torch.bool)
        mask[end_pos] = False

        src = flat[:-1][mask]
        dst = flat[1:][mask]

        if src.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty, M

        key = src * M + dst
        uniq, cnt = torch.unique(key, return_counts=True)

        src_u = (uniq // M).to(torch.long)
        dst_u = (uniq %  M).to(torch.long)
        cnt_u = cnt.to(torch.long)
        return src_u, dst_u, cnt_u, M

    def _build_topk_from_global_counter(self, src_u, dst_u, cnt_u, M: int, K: int):
        device = src_u.device

        if src_u.numel() == 0:
            top_idx = torch.full((M, K), -1, dtype=torch.int32, device=device)
            top_p = torch.zeros((M, K), dtype=torch.float32, device=device)
            top_idx[:, 0] = torch.arange(M, device=device, dtype=torch.int32)
            top_p[:, 0] = 1.0
            return top_idx, top_p

        order = torch.argsort(src_u)
        src_s = src_u[order]
        dst_s = dst_u[order]
        cnt_s = cnt_u[order].to(torch.float32)

        uniq_src, counts = torch.unique_consecutive(src_s, return_counts=True)
        num_uniq = uniq_src.numel()

        max_neighbors = int(counts.max().item())

        if max_neighbors <= K:
            top_idx = torch.full((M, K), -1, dtype=torch.int32, device=device)
            top_p = torch.zeros((M, K), dtype=torch.float32, device=device)

            starts = torch.zeros(num_uniq + 1, device=device, dtype=torch.long)
            starts[1:] = counts.cumsum(0)

            positions = torch.arange(src_s.numel(), device=device) - starts[:-1].repeat_interleave(counts)

            valid_mask = positions < K
            if valid_mask.any():
                valid_src = src_s[valid_mask]
                valid_pos = positions[valid_mask]
                valid_dst = dst_s[valid_mask]
                valid_cnt = cnt_s[valid_mask]

                row_sums = torch.zeros(M, device=device, dtype=torch.float32)
                row_sums.scatter_add_(0, src_s, cnt_s)

                valid_p = valid_cnt / row_sums[valid_src].clamp_min(1e-12)

                flat_idx = valid_src * K + valid_pos
                top_idx.view(-1)[flat_idx] = valid_dst.to(torch.int32)
                top_p.view(-1)[flat_idx] = valid_p
        else:
            top_idx = torch.full((M, K), -1, dtype=torch.int32, device=device)
            top_p = torch.zeros((M, K), dtype=torch.float32, device=device)

            starts = torch.zeros(num_uniq + 1, device=device, dtype=torch.long)
            starts[1:] = counts.cumsum(0)

            small_mask = counts <= K
            large_mask = ~small_mask

            if small_mask.any():
                small_indices = small_mask.nonzero(as_tuple=True)[0]
                for idx in small_indices:
                    s = int(starts[idx].item())
                    e = int(starts[idx + 1].item())
                    src = int(uniq_src[idx].item())
                    L = e - s
                    if L > 0:
                        seg_dst = dst_s[s:e]
                        seg_cnt = cnt_s[s:e]
                        seg_p = seg_cnt / seg_cnt.sum().clamp_min(1e-12)
                        top_idx[src, :L] = seg_dst.to(torch.int32)
                        top_p[src, :L] = seg_p

            if large_mask.any():
                large_indices = large_mask.nonzero(as_tuple=True)[0]
                for idx in large_indices:
                    s = int(starts[idx].item())
                    e = int(starts[idx + 1].item())
                    src = int(uniq_src[idx].item())

                    seg_cnt = cnt_s[s:e]
                    seg_dst = dst_s[s:e]

                    _, topk_idx = torch.topk(seg_cnt, k=K, largest=True)
                    sel_dst = seg_dst[topk_idx]
                    sel_cnt = seg_cnt[topk_idx]
                    sel_p = sel_cnt / sel_cnt.sum().clamp_min(1e-12)

                    top_idx[src, :K] = sel_dst.to(torch.int32)
                    top_p[src, :K] = sel_p

        empty = (top_p.sum(dim=1) <= 0)
        if empty.any():
            ridx = empty.nonzero(as_tuple=True)[0]
            top_idx[ridx, 0] = ridx.to(torch.int32)
            top_p[ridx, 0] = 1.0

        return top_idx, top_p

    def _prepare_sequences_on_device(self, all_user_sequences):
        num_users = len(all_user_sequences)
        if num_users == 0:
            return None

        all_seq_lengths = np.array([len(all_user_sequences.get(uid, [])) for uid in range(num_users)], dtype=np.int64)
        total_len = int(all_seq_lengths.sum())

        if total_len == 0:
            return None

        sequences_flat = np.empty(total_len, dtype=np.int64)
        pos = 0
        for uid in range(num_users):
            seq = all_user_sequences.get(uid, [])
            L = len(seq)
            if L > 0:
                sequences_flat[pos:pos + L] = seq
            pos += L

        offsets_np = np.zeros(num_users + 1, dtype=np.int64)
        offsets_np[1:] = np.cumsum(all_seq_lengths)

        sequences_on_device = {
            'flat': torch.from_numpy(sequences_flat).to(device=self.device, dtype=torch.long),
            'offsets': torch.from_numpy(offsets_np).to(device=self.device, dtype=torch.long),
        }
        return sequences_on_device

    def _create_transition_matrix_gpu(self, history_sequences: Dict[int, List[int]], num_items: int, max_k: int = 5) -> cpsp.csr_matrix:
        
        cp.cuda.Device(self.cupy_device_id).use()

        sequences = list(history_sequences.values())
        
        lengths = np.array([len(s) for s in sequences], dtype=np.int32)
        
        flat_items = np.concatenate(sequences).astype(np.int32)
        
        seq_ids = np.repeat(np.arange(len(sequences), dtype=np.int32), lengths)

        all_items_gpu = cp.array(flat_items)
        seq_ids_gpu = cp.array(seq_ids)
        
        del flat_items, seq_ids

        sources_list = []
        targets_list = []
        weights_list = []

        for k in range(1, max_k + 1):
            valid_mask = (seq_ids_gpu[:-k] == seq_ids_gpu[k:])
            
            if int(valid_mask.sum()) == 0:
                continue

            s = all_items_gpu[:-k][valid_mask]
            t = all_items_gpu[k:][valid_mask]
            
            sources_list.append(s)
            targets_list.append(t)
            
            w = cp.full(s.shape, 1.0 / k, dtype=cp.float32)
            weights_list.append(w)

        if not sources_list:
            return cpsp.csr_matrix((num_items, num_items), dtype=cp.float32)

        sources_final = cp.concatenate(sources_list)
        targets_final = cp.concatenate(targets_list)
        weights_final = cp.concatenate(weights_list)

        transitions_coo = cpsp.coo_matrix(
            (weights_final, (sources_final, targets_final)), 
            shape=(num_items, num_items)
        )
        
        transitions_csr = transitions_coo.tocsr()

        del all_items_gpu, seq_ids_gpu, sources_list, targets_list, weights_list
        del sources_final, targets_final, weights_final, transitions_coo
        cp.get_default_memory_pool().free_all_blocks()

        return transitions_csr

    def _randomized_svd_gpu(
        self,
        A: cupy_csr_matrix,
        k: int,
        n_iter: int = 5,
        omega_seed: Optional[int] = None,
    ):
        h, w = A.shape
        p = 20
        l = min(k + p, min(h, w))

        if omega_seed is None:
            Omega = cp.random.standard_normal((w, l), dtype=cp.float32)
        else:
            rng = cp.random.RandomState(int(omega_seed) % (2**32))
            Omega = rng.standard_normal((w, l), dtype=cp.float32)
        
        Y = A.dot(Omega) 
        
        for _ in range(n_iter):
            Q, _ = cp.linalg.qr(Y)
            Y = A.T.dot(Q)
            Q, _ = cp.linalg.qr(Y)
            Y = A.dot(Q)
        
        Q, _ = cp.linalg.qr(Y)
        
        B = A.T.dot(Q).T

        Uhat, S, Vt = cp.linalg.svd(B, full_matrices=False)
        
        U = Q.dot(Uhat)

        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        
        if cp.isnan(U_k).any() or cp.isnan(S_k).any() or cp.isnan(Vt_k).any():
            return None, None, None

        max_abs_cols = cp.argmax(cp.abs(U_k), axis=0)
        signs = cp.sign(U_k[max_abs_cols, cp.arange(U_k.shape[1])])
        signs[signs == 0] = 1.0
        
        U_k *= signs[None, :]
        Vt_k *= signs[:, None]
            
        return U_k, S_k, Vt_k

    def _compute_csr_svd_embeddings_gpu(
        self,
        matrix_gpu: cupy_csr_matrix,
        feature_dim: int,
        omega_seed: Optional[int] = None,
    ):
        cp.cuda.Device(self.cupy_device_id).use()
        matrix_gpu = matrix_gpu.copy()
        min_side = min(matrix_gpu.shape)
        k = min(feature_dim, min_side - 1)

        if k <= 0:
            return None, None

        try:
            U, S, Vt = self._randomized_svd_gpu(matrix_gpu, k=k, n_iter=5, omega_seed=omega_seed)
            
            if U is None:
                return None, None

            sqrt_S = cp.sqrt(S)
            u_cp = U * sqrt_S
            i_cp = Vt.T * sqrt_S

            if k < feature_dim:
                pad_width = feature_dim - k
                u_cp = cp.pad(u_cp, ((0, 0), (0, pad_width)), mode='constant')
                i_cp = cp.pad(i_cp, ((0, 0), (0, pad_width)), mode='constant')

            row_embeddings = torch.utils.dlpack.from_dlpack(u_cp.toDlpack()).to(self.device)
            col_embeddings = torch.utils.dlpack.from_dlpack(i_cp.toDlpack()).to(self.device)
            
            if torch.all(row_embeddings == 0) or torch.all(col_embeddings == 0):
                return None, None
            
            del U, S, Vt, u_cp, i_cp, sqrt_S, matrix_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            return row_embeddings, col_embeddings

        except Exception:
            cp.get_default_memory_pool().free_all_blocks()
            return None, None

    def _create_matrix_from_sequences(self, sequences: Dict[int, List[int]], num_users: int, num_items: int) -> torch.Tensor:
        total_interactions = sum(len(seq) for seq in sequences.values())

        if total_interactions == 0:
            return torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                (num_users, num_items)
            )

        users_np = np.empty(total_interactions, dtype=np.int64)
        items_np = np.empty(total_interactions, dtype=np.int64)

        pos = 0
        for user_idx, seq in sequences.items():
            if not seq:
                continue
            L = len(seq)
            users_np[pos:pos + L] = user_idx
            items_np[pos:pos + L] = seq
            pos += L

        indices = torch.from_numpy(np.stack([users_np[:pos], items_np[:pos]], axis=0)).to(
            device=self.device, dtype=torch.long
        )
        values = torch.ones(pos, dtype=torch.float32, device=self.device)

        new_sparse_matrix = torch.sparse_coo_tensor(indices, values, (num_users, num_items))

        return new_sparse_matrix.coalesce()

    def _apply_ppmi_transform(self, matrix_csr):
        matrix_coo = matrix_csr.astype(cp.float32).tocoo()
        
        total_sum = matrix_coo.data.sum()
        
        row_sums = cp.array(matrix_coo.sum(axis=1)).flatten()
        col_sums = cp.array(matrix_coo.sum(axis=0)).flatten()
        
        row_sums[row_sums == 0] = 1.0
        col_sums[col_sums == 0] = 1.0
        
        row_indices = matrix_coo.row
        col_indices = matrix_coo.col
        vals = matrix_coo.data
        
        numerator = vals * total_sum
        
        denominator = row_sums[row_indices] * col_sums[col_indices]
        
        ratio = numerator / denominator
        
        ratio = cp.maximum(ratio, 1e-12)
        
        ppmi_vals = cp.log(ratio)
        
        ppmi_vals = cp.maximum(ppmi_vals, 0.0)
        
        matrix_coo.data = ppmi_vals
        
        matrix_result = matrix_coo.tocsr()
        matrix_result.eliminate_zeros()
        
        return matrix_result
