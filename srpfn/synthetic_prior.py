from typing import Dict, List

import cupy as cp
import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.dlpack as dlpack

class SyntheticPrior:
    def _sample_power_law(self, xmin: int, xmax: int, alpha: float, size: int):
        ymax = 1 - (xmin / xmax) ** alpha
        y = np.random.uniform(0, ymax, size=size)
        samples = xmin * (1 - y) ** (-1. / alpha)
        return samples.astype(int)

    def _hierarchical_dcsbm_transition(self, N: int, directed=True):
        import graph_tool as gt
        from graph_tool.spectral import adjacency

        def adjust_sizes(sizes, target, min_val, rng):
            current_sum = int(sizes.sum())
            limit = 0
            while current_sum != target and limit < 50000:
                diff = target - current_sum
                step = int(np.sign(diff))
                if step > 0:
                    idx = int(rng.integers(0, len(sizes)))
                    sizes[idx] += 1
                else:
                    valid_idx = np.where(sizes > min_val)[0]
                    if valid_idx.size == 0:
                        break
                    idx = int(rng.choice(valid_idx))
                    sizes[idx] -= 1
                current_sum = int(sizes.sum())
                limit += 1
            return sizes

        try:
            rng = np.random.default_rng()

            avg_degree = int(rng.integers(32, 128))
            target_E = int(N * avg_degree)
            target_leak = float(rng.uniform(0.1, 0.5))
            w_micro_base = 10.0

            K_macro = int(np.clip(rng.integers(int(np.log2(N)), int(np.sqrt(N))), 5, 100))

            macro_alpha = float(rng.uniform(0.5, 1.5))
            macro_frac = rng.dirichlet(np.full(K_macro, macro_alpha))
            macro_sizes = np.maximum(8, np.round(macro_frac * N).astype(np.int64))
            macro_sizes = adjust_sizes(macro_sizes, N, 8, rng)

            macro_imp = rng.beta(2, 5, size=K_macro)
            macro_imp = (macro_imp - macro_imp.min()) / (macro_imp.max() - macro_imp.min() + 1e-6)

            block_sizes_list = []
            block_macro_of_list = []

            for m, M in enumerate(macro_sizes):
                M = int(M)
                base_res = 50.0 if macro_imp[m] > 0.6 else 100.0
                m_res = float(rng.uniform(base_res * 0.5, base_res * 1.5))

                m_cnt_target = max(1, int(round(M / m_res)))
                m_cnt = int(np.clip(rng.poisson(m_cnt_target), 1, max(1, M // 2)))

                if m_cnt <= 1:
                    sizes = np.array([M], dtype=np.int64)
                else:
                    m_alpha = float(rng.uniform(1.5, 3.0))
                    m_frac = rng.dirichlet(np.full(m_cnt, m_alpha))
                    sizes = np.maximum(2, np.round(m_frac * M).astype(np.int64))
                    sizes = adjust_sizes(sizes, M, 2, rng)

                block_sizes_list.extend(sizes.tolist())
                block_macro_of_list.extend([m] * len(sizes))

            block_sizes = np.array(block_sizes_list, dtype=np.int64)
            block_macro_of = np.array(block_macro_of_list, dtype=np.int64)
            B = int(len(block_sizes))
            b = np.repeat(np.arange(B, dtype=np.int64), block_sizes)

            def fix_sum(x: np.ndarray, T: int, rng: np.random.Generator,
                        min_val: int = 1, max_iter: int = 400000):
                s = int(x.sum())
                it = 0
                while s != T and it < max_iter:
                    if s < T:
                        i = int(rng.integers(0, x.size))
                        x[i] += 1
                        s += 1
                    else:
                        valid = np.where(x > min_val)[0]
                        if valid.size == 0:
                            break
                        i = int(rng.choice(valid))
                        x[i] -= 1
                        s -= 1
                    it += 1
                return x

            gamma = float(rng.uniform(1.6, 2.6))

            cap_pow = float(rng.uniform(0.55, 0.70))
            cap = int(max(32, min(N - 1, round(N ** cap_pow))))

            temp = float(rng.uniform(0.65, 0.90))

            x_out = np.clip(rng.zipf(gamma, N), 1, cap).astype(np.float64)
            x_in = np.clip(rng.zipf(gamma, N), 1, cap).astype(np.float64)

            if temp != 1.0:
                x_out = np.power(x_out, temp)
                x_in = np.power(x_in, temp)

            deg_floor = int(rng.integers(1, 3))
            out_deg = np.maximum(deg_floor, np.round(x_out / x_out.sum() * target_E)).astype(np.int64)
            in_deg = np.maximum(deg_floor, np.round(x_in  / x_in.sum()  * target_E)).astype(np.int64)

            deg_cap = int(max(deg_floor, min(N - 1, round(cap * rng.uniform(0.8, 1.2)))))
            out_deg = np.minimum(out_deg, deg_cap)
            in_deg  = np.minimum(in_deg,  deg_cap)

            out_deg = fix_sum(out_deg, target_E, rng, min_val=deg_floor)
            in_deg = fix_sum(in_deg,  target_E, rng, min_val=deg_floor)

            w_macro_base_vec = 10.0 * (10 ** rng.uniform(-2.5, -1.0, size=K_macro))
            w_macro_vec = w_macro_base_vec * (1.0 + macro_imp)

            nr = block_sizes.astype(np.float64)
            macro_sz = np.bincount(block_macro_of, weights=nr, minlength=K_macro)
            micro_sq_per_macro = np.bincount(block_macro_of, weights=nr * nr, minlength=K_macro)
            S_macro_off_per_macro = np.maximum(0.0, macro_sz * macro_sz - micro_sq_per_macro)

            S_micro = float((nr * nr).sum())
            S_cross = float(max(1e-12, N * N - float((macro_sz * macro_sz).sum())))

            internal_sum = w_micro_base * S_micro + float((w_macro_vec * S_macro_off_per_macro).sum())
            w_cross_target_avg = (internal_sum * target_leak) / max((1.0 - target_leak) * S_cross, 1e-12)

            M_rel = np.outer(macro_imp, macro_imp)
            off = ~np.eye(K_macro, dtype=bool)

            M_rel_mean_off = float(M_rel[off].mean() + 1e-9)

            M_multiplier = 0.5 + (M_rel / M_rel_mean_off)
            M_multiplier /= float(M_multiplier[off].mean() + 1e-9)

            M_conn_strength = w_cross_target_avg * M_multiplier

            limit_val = float(w_macro_vec.min() * 0.95)
            M_conn_strength = np.clip(M_conn_strength, 1e-12, limit_val)

            macro_idx_mat = block_macro_of[:, None]
            macro_idx_mat_T = block_macro_of[None, :]

            W = M_conn_strength[macro_idx_mat, macro_idx_mat_T].astype(np.float64)

            for m_id in range(K_macro):
                idx = np.where(block_macro_of == m_id)[0]
                if idx.size > 0:
                    W[np.ix_(idx, idx)] = w_macro_vec[m_id]

            np.fill_diagonal(W, w_micro_base)

            ers = W * (nr[:, None] * nr[None, :])

            current_expected_E = float(ers.sum())
            scaling = float(target_E) / max(current_expected_E, 1e-12)
            ers *= scaling

            g = gt.generation.generate_sbm(
                b=b, probs=ers, out_degs=out_deg, in_degs=in_deg,
                directed=directed, condensed=False
            )

            A_csr = adjacency(g).tocsr()
            row_sum = np.asarray(A_csr.sum(axis=1)).ravel().astype(np.float64)
            inv = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
            P = (sp.diags(inv) @ A_csr).tocsr()

            return P

        except Exception:
            return None

    def _generate_sequences(
        self,
        transition_matrix,
        num_users: int,
        num_items: int,
        batch_size: int = 4096,
        pl_alpha_range: tuple = (1.2, 2.5),
        max_seq_len_range: tuple = (100, 2000),
        tau: float = 1.0,
        max_lag: int = 16,
        decay_range: tuple = (0.05, 0.98),
    ):
        rng = np.random.default_rng()
        device = self.device if isinstance(self.device, torch.device) else torch.device("cuda")
        if device.type != "cuda":
            device = torch.device("cuda")

        N = int(num_items)
        max_lag_eff = int(min(max(1, max_lag), 5))

        if sp.issparse(transition_matrix):
            indptr_cp = cp.asarray(transition_matrix.indptr,  dtype=cp.int64)
            indices_cp = cp.asarray(transition_matrix.indices, dtype=cp.int64)
            data_cp = cp.asarray(transition_matrix.data,    dtype=cp.float32)
        else:
            indptr_cp = transition_matrix.indptr.astype(cp.int64, copy=False)
            indices_cp = transition_matrix.indices.astype(cp.int64, copy=False)
            data_cp = transition_matrix.data.astype(cp.float32, copy=False)

        indptr_t = dlpack.from_dlpack(indptr_cp.toDlpack()).to(device=device, dtype=torch.long)
        indices_t = dlpack.from_dlpack(indices_cp.toDlpack()).to(device=device, dtype=torch.long)
        data_t = dlpack.from_dlpack(data_cp.toDlpack()).to(device=device, dtype=torch.float32)

        del indptr_cp, indices_cp, data_cp
        cp.get_default_memory_pool().free_all_blocks()

        last_valid = int(indices_t.numel()) - 1
        last_valid_t = torch.tensor(last_valid, device=device, dtype=torch.long)

        row_degrees = indptr_t[1:] - indptr_t[:-1]
        global_max_deg = int(row_degrees.max().item())
        del row_degrees

        ar_buffer = torch.arange(global_max_deg, device=device, dtype=torch.long)

        pl_xmin = 5
        
        pl_alpha = float(rng.uniform(pl_alpha_range[0], pl_alpha_range[1]))
        max_seq_len = int(rng.integers(int(max_seq_len_range[0]), int(max_seq_len_range[1]) + 1))
        seq_lengths = self._sample_power_law(pl_xmin, max_seq_len, pl_alpha, size=num_users).astype(np.int32)

        row_degrees = (indptr_t[1:] - indptr_t[:-1]).to(torch.float32)
        if row_degrees.sum() > 0:
            start_probs = row_degrees / row_degrees.sum()
            user_seeds_t = torch.multinomial(start_probs, num_users, replacement=True)
            user_seeds = user_seeds_t.cpu().numpy()
        else:
            user_seeds = rng.integers(0, N, size=num_users, dtype=np.int64)

        dmin, dmax = float(decay_range[0]), float(decay_range[1])
        dmin = max(dmin, 1e-6)
        dmax = min(dmax, 0.999999)

        rho_graph = float(rng.uniform(dmin, dmax))
        
        log_rho = np.log(max(rho_graph, 1e-6))

        all_sequences = {}

        for batch_start in range(0, num_users, batch_size):
            batch_end = min(batch_start + batch_size, num_users)
            bs = batch_end - batch_start

            lens = seq_lengths[batch_start:batch_end]
            maxL = int(lens.max())
            lens_t = torch.from_numpy(lens).to(device=device, dtype=torch.long)

            current = torch.from_numpy(user_seeds[batch_start:batch_end]).to(device=device, dtype=torch.long)
            seq = torch.empty((bs, maxL), device=device, dtype=torch.long)
            seq[:, 0] = current

            hist = torch.empty((max_lag_eff, bs), device=device, dtype=torch.long)
            hist[0] = current
            if max_lag_eff > 1:
                hist[1:] = current.unsqueeze(0)

            active_mask = torch.ones(bs, device=device, dtype=torch.bool)

            for t in range(1, maxL):
                active_mask &= (t < lens_t)
                if not active_mask.any():
                    break

                active = active_mask.nonzero(as_tuple=True)[0]
                A = active.numel()

                u = torch.rand(A, device=device).clamp_min(1e-8)
                L = (1 + torch.floor(torch.log(u) / log_rho)).clamp_(1, max_lag_eff).to(torch.long)

                hist_active = hist[:, active]
                src = hist_active.gather(0, (L - 1).view(1, A)).squeeze(0)

                row_st = indptr_t[src]
                row_ed = indptr_t[src + 1]
                deg = row_ed - row_st

                if global_max_deg <= 0:
                    nxt = torch.randint(0, N, (A,), device=device, dtype=torch.long)
                else:
                    actual_max_deg = min(global_max_deg, int(deg.max().item()) if A < 1000 else global_max_deg)
                    ar = ar_buffer[:actual_max_deg]
                    offs = row_st.unsqueeze(1) + ar.unsqueeze(0)
                    mask = ar.unsqueeze(0) < deg.unsqueeze(1)

                    offs = torch.where(mask, offs, last_valid_t)
                    nbr = indices_t[offs]
                    prob = data_t[offs]

                    logit = torch.log(prob.clamp_min(1e-12))
                    logit = logit.masked_fill(~mask, -1e10)

                    if tau != 1.0:
                        logit = logit / float(tau)

                    probs = torch.softmax(logit, dim=1)
                    idx = torch.multinomial(probs, 1).squeeze(1)
                    nxt = nbr[torch.arange(A, device=device), idx].to(torch.long)

                current[active] = nxt
                seq[active, t] = nxt

                if max_lag_eff > 1:
                    hist[1:, active] = hist[:-1, active]
                hist[0, active] = current[active]

            seq_np = seq.cpu().numpy()
            lens_np = lens
            for i in range(bs):
                all_sequences[batch_start + i] = seq_np[i, :lens_np[i]].tolist()

            del seq, current, hist, lens_t, active_mask

        del indptr_t, indices_t, data_t, ar_buffer
        torch.cuda.empty_cache()

        return all_sequences

    def _apply_k_core_filtering(self, sequences: Dict[int, List[int]], k: int) -> Dict[int, List[int]]:
        if not sequences:
            return {}

        all_items = []
        user_lengths = {}

        for u, seq in sequences.items():
            L = len(seq)
            user_lengths[u] = L
            all_items.extend(seq)

        if not all_items:
            return {}

        all_items_np = np.array(all_items, dtype=np.int64)

        max_item = all_items_np.max() + 1
        item_counts = np.bincount(all_items_np, minlength=max_item)

        valid_items_mask = item_counts >= k

        valid_users = {u for u, L in user_lengths.items() if L >= k}

        filtered = {}
        for u, seq in sequences.items():
            if u not in valid_users:
                continue
            seq_np = np.array(seq, dtype=np.int64)
            keep_mask = valid_items_mask[seq_np]
            kept = seq_np[keep_mask].tolist()
            if kept:
                filtered[u] = kept

        return filtered
