import math
import random

import cupy as cp
import numpy as np
import torch
from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
from torch.utils.data import IterableDataset

from srpfn.batch_builder import BatchBuilder
from srpfn.features import FeatureBuilder
from srpfn.synthetic_prior import SyntheticPrior

class RecIterableDataset(SyntheticPrior, FeatureBuilder, BatchBuilder, IterableDataset):
    def __init__(
        self,
        max_support_set_size,
        num_candidates,
        feature_dim,
        batch_size,
        steps_per_epoch,
        max_history_len,
        device='cpu',
        yield_on_cpu=False,
        scoring_mode='candidate',
        prior_seed_base: int = 0,
    ):
        super().__init__()

        self.max_support_set_size = max_support_set_size
        self.num_candidates = num_candidates
        self.num_negatives = num_candidates - 1
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.max_history_len = max_history_len

        self.device = torch.device(device) if isinstance(device, str) else device

        self.yield_on_cpu = yield_on_cpu
        self.scoring_mode = scoring_mode
        if self.scoring_mode not in {"candidate", "full_item"}:
            raise ValueError(f"Unsupported scoring_mode={self.scoring_mode}")

        if self.device.type == 'cuda':
            self.cupy_device_id = self.device.index if self.device.index is not None else 0
        else:
            self.cupy_device_id = 0

        self.steps_per_epoch = steps_per_epoch
        self.steps_yielded_this_epoch = 0
        self.prior_seed_base = int(prior_seed_base)
        self.current_cycle_seed = int(prior_seed_base)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            per_worker_steps = self.steps_per_epoch
        else:
            per_worker_steps = int(math.ceil(self.steps_per_epoch / float(worker_info.num_workers)))
            
            worker_seed = (torch.initial_seed() + worker_info.id) % (2**32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            
        self.steps_yielded_this_epoch = 0

        while self.steps_yielded_this_epoch < per_worker_steps:
            resources = {}
            try:
                M = math.ceil(10000 * random.uniform(1, 5))
                if M < self.feature_dim:
                    continue

                try:
                    transition_matrix = self._hierarchical_dcsbm_transition(
                        N=M,
                        directed=True,
                    )
                    
                except Exception:
                    continue
                if transition_matrix is None:
                    continue
                
                multiplier = random.uniform(0.5, 2.0)
                num_synthetic_users = int(M * multiplier)
                
                try:
                    user_sequences = self._generate_sequences(
                        transition_matrix, num_synthetic_users, M
                    )
                    
                except Exception:
                    continue
                    
                if not user_sequences: 
                    continue

                temp_uids = list(user_sequences.keys())
                seq_lengths = np.array([len(user_sequences[uid]) for uid in temp_uids], dtype=np.int64)
                total_len = int(seq_lengths.sum())

                all_items_flat = np.empty(total_len, dtype=np.int64)
                seq_offsets = np.zeros(len(temp_uids) + 1, dtype=np.int64)
                seq_offsets[1:] = np.cumsum(seq_lengths)

                pos = 0
                for i, uid in enumerate(temp_uids):
                    seq = user_sequences[uid]
                    L = len(seq)
                    all_items_flat[pos:pos + L] = seq
                    pos += L

                flat_tensor = torch.from_numpy(all_items_flat).to(device=self.device, dtype=torch.long)

                perm_mapping = torch.randperm(M, device=self.device)
                flat_tensor = perm_mapping[flat_tensor]

                flat_np = flat_tensor.cpu().numpy()
                all_user_sequences = {}

                for i, uid in enumerate(temp_uids):
                    start, end = int(seq_offsets[i]), int(seq_offsets[i + 1])
                    if end - start >= 2:
                        all_user_sequences[uid] = flat_np[start:end].tolist()

                core_filtered_sequences_unmapped = self._apply_k_core_filtering(all_user_sequences, k=5)
                
                num_users_after_filter = len(core_filtered_sequences_unmapped)
                
                if num_users_after_filter == 0:
                     continue

                final_item_set = {item for seq in core_filtered_sequences_unmapped.values() for item in seq}
                num_items_after_filter = len(final_item_set)

                if num_users_after_filter < 5000 or num_items_after_filter < 5000:
                    continue

                final_user_ids = sorted(list(core_filtered_sequences_unmapped.keys()))
                user_remap = {old_id: new_id for new_id, old_id in enumerate(final_user_ids)}

                final_sorted_items = sorted(list(final_item_set))
                max_old_item = max(final_sorted_items) + 1
                item_remap_arr = np.full(max_old_item, -1, dtype=np.int64)
                for new_id, old_id in enumerate(final_sorted_items):
                    item_remap_arr[old_id] = new_id

                final_sequences = {}
                for old_user_id, seq in core_filtered_sequences_unmapped.items():
                    new_user_id = user_remap[old_user_id]
                    seq_np = np.array(seq, dtype=np.int64)
                    remapped_seq = item_remap_arr[seq_np].tolist()
                    final_sequences[new_user_id] = remapped_seq

                M = len(final_item_set)
                num_synthetic_users = len(final_sequences)
                
                histories = {
                    uid: (final_sequences[uid][:-1] if len(final_sequences[uid]) > 1 else [])
                    for uid in range(num_synthetic_users)
                }
                
                transition_matrix_gpu = self._create_transition_matrix_gpu(histories, M)
                
                transition_matrix_gpu = self._apply_ppmi_transform(transition_matrix_gpu)
                
                resources['trans_r'], resources['trans_c'] = self._compute_csr_svd_embeddings_gpu(
                    transition_matrix_gpu, self.feature_dim
                )
                
                if resources['trans_r'] is None:
                    continue
                    
                del transition_matrix_gpu

                user_item_matrix_torch = self._create_matrix_from_sequences(
                    histories, num_users=num_synthetic_users, num_items=M
                )
                
                coo = user_item_matrix_torch.coalesce()
                                
                cupy_dev_str = f'cuda:{self.cupy_device_id}'
                
                r_idx_cuda = coo.indices()[0].to(device=cupy_dev_str, dtype=torch.int32)
                c_idx_cuda = coo.indices()[1].to(device=cupy_dev_str, dtype=torch.int32)
                vals_cuda  = coo.values().to(device=cupy_dev_str, dtype=torch.float32)

                rows_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(r_idx_cuda))
                cols_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(c_idx_cuda))
                vals_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(vals_cuda))

                bipartite_csr = cupy_csr_matrix((vals_cp, (rows_cp, cols_cp)), shape=coo.shape)
                
                del user_item_matrix_torch, coo, r_idx_cuda, c_idx_cuda, vals_cuda
                         
                bipartite_csr = self._apply_ppmi_transform(bipartite_csr)
                            
                resources['user_cf'], resources['item_cf'] = self._compute_csr_svd_embeddings_gpu(
                    bipartite_csr, self.feature_dim
                )
                del bipartite_csr, rows_cp, cols_cp, vals_cp
                
                if resources['user_cf'] is None:
                    continue

                valid_user_ids = list(final_sequences.keys())
                random.shuffle(valid_user_ids)
                
                remaining_steps = self.steps_per_epoch - self.steps_yielded_this_epoch
                if remaining_steps <= 0:
                    break

                max_prompts = min(2 * self.batch_size * remaining_steps, len(valid_user_ids))
                queries_to_process = valid_user_ids[:max_prompts]

                sequences_full = self._prepare_sequences_on_device(final_sequences)
                sequences_obs  = self._prepare_sequences_on_device(histories)

                src_u, dst_u, cnt_u, M = self._build_global_transition_counter(sequences_obs=sequences_obs)
                top_idx, top_p = self._build_topk_from_global_counter(src_u, dst_u, cnt_u, M=M, K=512)

                embeddings = {
                    'user_cf': resources['user_cf'],
                    'item_cf': resources['item_cf'],
                    'trans_r': resources['trans_r'],
                    'trans_c': resources['trans_c'],
                }
                
                for i in range(0, len(queries_to_process), self.batch_size):
                    if self.steps_yielded_this_epoch >= per_worker_steps:
                        break

                    chunk = queries_to_process[i : i + self.batch_size]
                    if not chunk: continue

                    try:
                        final_batch = self._generate_and_create_batch(
                            query_uids_batch=torch.tensor(chunk, device=self.device),
                            sequences_obs=sequences_obs,
                            sequences_full=sequences_full,
                            top_idx=top_idx,
                            top_p=top_p,
                            M=M,
                            embeddings=embeddings,
                            support_set_size=random.randint(1, self.max_support_set_size),
                        )

                        if final_batch is None: continue

                        final_batch_cpu = {
                            k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                            for k, v in final_batch.items()
                        }

                        del final_batch

                        yield final_batch_cpu
                        self.steps_yielded_this_epoch += 1

                    except Exception:
                        continue

                del sequences_full, sequences_obs
                del top_idx, top_p, src_u, dst_u, cnt_u
                del embeddings
                torch.cuda.empty_cache()
                    
            except Exception:
                continue
            
            finally:
                for var_name in ['sequences_full', 'sequences_obs',
                                 'top_idx', 'top_p', 'src_u', 'dst_u', 'cnt_u',
                                 'embeddings', 'flat_tensor', 'perm_mapping']:
                    if var_name in locals():
                        del locals()[var_name]

                if 'resources' in locals():
                    for key in list(resources.keys()):
                        del resources[key]
                    del resources

                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
