import argparse
import json
import logging
import math
import os
import random
import time
from collections import Counter
from typing import Dict, List, Optional

import cupy as cp
import numpy as np
import torch
import torch.nn as nn
from torch.backends.cuda import sdp_kernel
from scipy.sparse import lil_matrix
from tqdm import tqdm

from srpfn.dataset import RecIterableDataset
from transformer import SRPFN
from utils import get_encoder_generator

PROCESS_START_TIME = time.time()
TQDM_FILE = None

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu_device', default='cuda', type=str)
    parser.add_argument('--log_file', type=str, default=None)
    return parser.parse_args()

def _seed_rngs(seed: int):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        cp.random.seed(seed % (2**32))
    except Exception:
        pass

def _derive_batch_seed(base_seed: int, tag: str, batch_idx: int) -> int:
    tag_salt = sum((i + 1) * ord(ch) for i, ch in enumerate(tag))
    return int((int(base_seed) + tag_salt + int(batch_idx) * 1_000_003) % (2**31 - 1))

def set_seed(seed: int, deterministic: bool = True):
    _seed_rngs(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except AttributeError:
            pass

def _format_duration(seconds: float) -> str:
    total = int(max(0, round(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def _resolve_eval_data_paths(eval_config: dict) -> dict:
    dataset = eval_config.get("dataset")
    if not dataset:
        raise ValueError("eval_config must include 'dataset'.")

    resolved = dict(eval_config)
    resolved.setdefault("seq_file", os.path.join("data", f"{dataset}.txt"))
    resolved.setdefault("cand_file", os.path.join("data", f"{dataset}_cand.txt"))
    return resolved

def process_data(
    seq_file: str,
    cand_file: str,
    dataset: str,
    logger: Optional[logging.Logger] = None,
):
    """
    S3Rec 형식의 데이터 파일을 읽어서 SR-PFN 평가용 데이터로 변환하고 메모리로 반환

    입력 파일 형식:
      - seq_file ({dataset}.txt): "user_id item1 item2 ... itemN" (시간순 정렬, 1-indexed)
      - cand_file ({dataset}_cand.txt): "user_id neg1 neg2 ... neg99" (99 negatives)

    반환:
      - eval_sequences: Dict[int, List[int]]
      - candidates: Dict[int, List[int]]
      - interaction_matrix: csr_matrix
      - item_popularity: Dict[int, int]
      - stats: Dict[str, Any]
    """
    log = logger or logging.getLogger(__name__)

    log.info(f"Loading sequences from {seq_file}")
    eval_sequences: Dict[int, List[int]] = {}
    item_set = set()

    with open(seq_file, 'r') as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split(' ')
            items = [int(x) - 1 for x in parts[1:]]
            eval_sequences[line_idx] = items
            item_set.update(items)

    num_users = len(eval_sequences)
    max_item = max(item_set)
    num_items = max_item + 1

    log.info(f"  Loaded {num_users} users, num_items={num_items} (max_item={max_item})")

    log.info(f"Loading candidates from {cand_file}")
    candidates: Dict[int, List[int]] = {}

    with open(cand_file, 'r') as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split(' ')
            neg_items = [int(x) - 1 for x in parts[1:]]
            candidates[line_idx] = neg_items

    assert len(candidates) == num_users, \
        f"Mismatch: sequences has {num_users} users, candidates has {len(candidates)}"
    log.info(f"  Loaded {len(candidates[0])} negative candidates per user")

    log.info("Building interaction matrix (excluding test items)...")
    interaction_matrix = lil_matrix((num_users, num_items), dtype=np.float32)

    for user_idx, seq in tqdm(eval_sequences.items(), desc="Building matrix", file=TQDM_FILE):
        for item_id in seq[:-1]:
            interaction_matrix[user_idx, item_id] = 1.0

    log.info("Computing item popularity (train only)...")
    event_counts = Counter()
    for seq in eval_sequences.values():
        for item_id in seq[:-1]:
            event_counts[item_id] += 1

    popularity_dict = {i: int(event_counts.get(i, 0)) for i in range(num_items)}

    sorted_items = sorted(popularity_dict.items(), key=lambda x: x[1], reverse=True)
    n_items = len(sorted_items)
    head_boundary = int(n_items * 0.2)
    torso_boundary = int(n_items * 0.8)

    head_set = {item_id for item_id, _ in sorted_items[:head_boundary]}
    torso_set = {item_id for item_id, _ in sorted_items[head_boundary:torso_boundary]}
    tail_set = {item_id for item_id, _ in sorted_items[torso_boundary:]}

    head_users = torso_users = tail_users = 0
    for seq in eval_sequences.values():
        test_item = seq[-1]
        if test_item in head_set:
            head_users += 1
        elif test_item in torso_set:
            torso_users += 1
        else:
            tail_users += 1

    log.info(f"  Test item distribution: head={head_users}, torso={torso_users}, tail={tail_users}")

    seq_lengths = [len(seq) for seq in eval_sequences.values()]
    train_interactions = sum(len(seq) - 1 for seq in eval_sequences.values())

    stats = {
        'dataset': dataset,
        'num_users': num_users,
        'num_items': num_items,
        'max_item_id': max_item,
        'total_interactions': sum(seq_lengths),
        'train_interactions': train_interactions,
        'avg_seq_length': float(np.mean(seq_lengths)),
        'min_seq_length': int(np.min(seq_lengths)),
        'max_seq_length': int(np.max(seq_lengths)),
        'num_candidates': len(candidates[0]),
        'sparsity': float(1 - (train_interactions / (num_users * num_items))),
        'head_users': head_users,
        'torso_users': torso_users,
        'tail_users': tail_users,
    }

    log.info(f"  Statistics: users={num_users}, items={num_items}, train_interactions={train_interactions}")
    log.info(f"  Avg seq length: {stats['avg_seq_length']:.2f}, Sparsity: {stats['sparsity']*100:.2f}%")

    log.info(f"Data processing for {dataset} completed successfully.")
    return {
        "eval_sequences": eval_sequences,
        "candidates": candidates,
        "interaction_matrix": interaction_matrix.tocsr(),
        "item_popularity": popularity_dict,
        "stats": stats,
    }

def load_pretrained_model(checkpoint_path: str, device: str = 'cuda', logger=None):
    log = logger or logging.getLogger(__name__)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'config' in checkpoint:
            config = checkpoint['config']
        elif 'pretrained_config' in checkpoint:
            config = checkpoint['pretrained_config']
            log.info("Loaded fine-tuned model checkpoint")
        else:
            raise KeyError("Checkpoint has neither 'config' nor 'pretrained_config'")
        
        log.info("Reconstructing model from saved config...")
        model_params = config['model']
        data_params = config['data']
        log.info(f"Num candidates: {data_params['num_candidates']}")
        encoder_creator = get_encoder_generator()

        model = SRPFN(
            feature_dim=data_params['feature_dim'],
            emsize=model_params['emsize'],
            interaction_encoder=encoder_creator,
            nhead=model_params['nhead'],
            nhid=model_params['nhid_factor'] * model_params['emsize'],
            nlayers=model_params['nlayers'],
            max_history_len=data_params['max_history_len'],
            dropout=model_params['dropout'],
            activation=model_params['activation'],
            input_normalization=model_params['input_normalization']
        )
        
        state_key = 'ema_model_state_dict' if 'ema_model_state_dict' in checkpoint else 'model_state_dict'
        model_state_dict = checkpoint[state_key]

        has_module_prefix = any(k.startswith('module.') for k in model_state_dict.keys())
        if has_module_prefix:
            new_state_dict = {k[len("module."):]: v for k, v in model_state_dict.items()}
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(model_state_dict)

        model.to(device)
        model.eval()
        
        return model, config

    except Exception as e:
        log.error(f"Error loading model from {checkpoint_path}: {e}", exc_info=True)
        raise


def generate_data(
    model,
    model_config: dict,
    user_sequences: Dict[int, List[int]],
    num_users: int,
    num_items: int,
    device: str = 'cuda',
    batch_size: int = 32,
    max_eval_users: Optional[int] = None,
    support_set_size: int = 4,
    predefined_candidates: Optional[Dict[int, List[int]]] = None,
    mode: str = "cand",
    logger=None,
    progress_file=None,
):
    log = logger or logging.getLogger(__name__)
    log.info("="*50 + "\nPHASE 1: GENERATE INFERENCE DATA \n" + "="*50)
    log.info(f"support set size : {support_set_size}")
    log.info(f"eval mode : {mode}")
    if mode not in ("cand", "full"):
        raise ValueError(f"Unsupported eval mode: {mode}")
    scoring_mode = "full_item" if mode == "full" else "candidate"
    if predefined_candidates is not None:
        if mode == "full":
            log.info("Predefined candidates are ignored in full-item ranking mode.")
        else:
            log.info(f"Using predefined candidates: {len(predefined_candidates[0])} negatives + 1 positive = {len(predefined_candidates[0]) + 1} total")

    data_params = model_config['data']
    eval_seed = int(model_config.get("eval_seed", model_config.get("seed", 42)))
    feature_dim = data_params['feature_dim']
    max_history_len = data_params.get('max_history_len', 50)

    torch_device = torch.device(device) if not isinstance(device, torch.device) else device

    builder = RecIterableDataset(
        max_support_set_size=support_set_size,
        num_candidates=data_params['num_candidates'],
        feature_dim=feature_dim,
        batch_size=batch_size,
        steps_per_epoch=1,
        max_history_len=max_history_len,
        device=torch_device,
        yield_on_cpu=False,
        scoring_mode=scoring_mode,
        prior_seed_base=eval_seed,
    )
    builder.current_cycle_seed = eval_seed
    prior_state = builder.build_prior_state_from_sequences(
        user_sequences=user_sequences,
        num_users=num_users,
        num_items=num_items,
        feature_dim=feature_dim,
        seed=eval_seed,
        full_item=(mode == "full"),
    )

    overall_pool = [uid for uid, seq in user_sequences.items() if len(seq) >= 2]
    if max_eval_users is None or max_eval_users <= 0:
        overall_uids = overall_pool
    else:
        overall_max = min(max_eval_users, len(overall_pool))
        overall_uids = random.sample(overall_pool, overall_max) if len(overall_pool) > overall_max else overall_pool
    log.info(f"Overall={len(overall_uids)}")

    def _build_subdataset(selected_uids: List[int], tag: str):
        return builder.build_batches_for_users(
            selected_uids=selected_uids,
            prior_state=prior_state,
            user_sequences=user_sequences,
            num_users=num_users,
            batch_size=batch_size,
            support_set_size=support_set_size,
            predefined_candidates=predefined_candidates,
            mode=mode,
            seed_callback=lambda batch_idx: _seed_rngs(_derive_batch_seed(eval_seed, tag, batch_idx)),
            progress_desc=f"Generating {tag.lower()} batches",
            progress_file=progress_file,
        )

    overall_batches = _build_subdataset(overall_uids, tag="OVERALL")
    return {
        "overall": overall_batches,
    }

def inference(
    model: nn.Module,
    model_config: dict,
    all_batches: List[dict],
    item_popularity_groups: Dict[str, set],
    device: str = 'cuda',
    data_tag: str = "DATA",
    mode: str = "cand",
    logger=None,
    progress_file=None,
):
    log = logger or logging.getLogger(__name__)
    log.info("="*50 + "\nPHASE 2: INFERENCE\n" + "="*50)
    model.eval()

    if all_batches is None:
        log.error(f"[{data_tag}] Inference batches are missing.")
        return None
    log.info(f"[{data_tag}] Using {len(all_batches)} in-memory batches.")
    if len(all_batches) == 0:
        log.warning("No users were successfully evaluated.")
        return 0.0, 0.0, 0.0

    metrics = {
        'all':   {'hits': 0, 'hits5': 0, 'hits10': 0, 'hits20': 0, 'ndcg5': 0.0, 'ndcg10': 0.0, 'ndcg20': 0.0, 'count': 0},
        'head':  {'hits': 0, 'hits5': 0, 'hits10': 0, 'hits20': 0, 'ndcg5': 0.0, 'ndcg10': 0.0, 'ndcg20': 0.0, 'count': 0},
        'torso': {'hits': 0, 'hits5': 0, 'hits10': 0, 'hits20': 0, 'ndcg5': 0.0, 'ndcg10': 0.0, 'ndcg20': 0.0, 'count': 0},
        'tail':  {'hits': 0, 'hits5': 0, 'hits10': 0, 'hits20': 0, 'ndcg5': 0.0, 'ndcg10': 0.0, 'ndcg20': 0.0, 'count': 0}
    }

    inference_start_time = time.time()
    
    with torch.no_grad():
        for batch_cpu in tqdm(
            all_batches,
            desc=f"Inference {data_tag.lower()}",
            total=len(all_batches),
            unit="batch",
            file=progress_file,
        ):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_cpu.items()}

            interaction_vectors = batch["interaction_vectors"]
            position_ids = batch["position_ids"]
            attention_mask = batch["attention_mask"]

            labels = batch["labels"]
            answer_positions = batch["answer_positions"]
            
            support_interaction_vecs = batch["support_interaction_vecs"]
            support_valid_mask = batch["support_valid_mask"]

            if mode == "full":
                forward_kwargs = dict(
                    query_user_emb=batch["query_user_emb"],
                    query_prev_item_emb=batch["query_prev_item_emb"],
                    query_prev_trans_emb=batch["query_prev_trans_emb"],
                    all_item_cf=batch["all_item_cf"],
                    all_trans_c=batch["all_trans_c"],
                    num_items=int(batch["num_items"]),
                    item_chunk_size=model_config.get("model", {}).get("item_chunk_size", 1024),
                    full_item_temperature=model_config.get("model", {}).get("full_item_temperature", 0.07),
                )
                candidate_counts = None
                candidates = None
            else:
                candidate_counts = batch["candidate_counts"]
                candidate_interaction_vecs = batch["candidate_interaction_vecs"]
                candidates = batch.get("candidates", None)
                if candidates is None:
                    candidates = batch.get("candidate_items", None)
                if candidates is None:
                    raise KeyError("Batch is missing 'candidates' (or 'candidate_items').")
                forward_kwargs = dict(
                    candidate_interaction_vecs=candidate_interaction_vecs,
                    candidate_counts=candidate_counts,
                )

            with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                output = model(
                    interaction_vectors=interaction_vectors,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    answer_positions=answer_positions,
                    support_interaction_vecs=support_interaction_vecs,
                    support_valid_mask=support_valid_mask,
                    return_logits=True,
                    **forward_kwargs,
                )

                logits = output["logits"]

            if mode == "full":
                valid = labels.ge(0) & labels.lt(int(batch["num_items"]))
            else:
                valid = labels.ge(0)
                if candidate_counts is not None:
                    valid = valid & labels.lt(candidate_counts)

            valid_idx = valid.nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue

            C = logits.size(1)

            predicted_indices = torch.argmax(logits, dim=1)
            top5_indices = torch.topk(logits, min(5, C), dim=1).indices
            top10_indices = torch.topk(logits, min(10, C), dim=1).indices
            top20_indices = torch.topk(logits, min(20, C), dim=1).indices

            if mode == "full":
                true_item_ids = labels[valid]
                pred_item_ids = predicted_indices[valid]
                top5_item_ids = top5_indices[valid]
                top10_item_ids = top10_indices[valid]
                top20_item_ids = top20_indices[valid]
            else:
                true_item_ids = candidates[valid].gather(1, labels[valid].unsqueeze(1)).squeeze(1)
                pred_item_ids = candidates[valid].gather(1, predicted_indices[valid].unsqueeze(1)).squeeze(1)
                top5_item_ids = candidates[valid].gather(1, top5_indices[valid])
                top10_item_ids = candidates[valid].gather(1, top10_indices[valid])
                top20_item_ids = candidates[valid].gather(1, top20_indices[valid])

            true_item_ids_cpu = true_item_ids.cpu()
            pred_item_ids_cpu = pred_item_ids.cpu()
            top5_item_ids_cpu = top5_item_ids.cpu()
            top10_item_ids_cpu = top10_item_ids.cpu()
            top20_item_ids_cpu = top20_item_ids.cpu()

            for b in range(true_item_ids_cpu.size(0)):
                true_id = int(true_item_ids_cpu[b].item())
                pred_id = int(pred_item_ids_cpu[b].item())
                top5_ids = top5_item_ids_cpu[b].tolist()
                top10_ids = top10_item_ids_cpu[b].tolist()
                top20_ids = top20_item_ids_cpu[b].tolist()

                if true_id in item_popularity_groups['head']: grp = 'head'
                elif true_id in item_popularity_groups['torso']: grp = 'torso'
                else: grp = 'tail'

                metrics['all']['count'] += 1
                metrics[grp]['count'] += 1

                if pred_id == true_id:
                    metrics['all']['hits'] += 1
                    metrics[grp]['hits'] += 1

                if true_id in top5_ids:
                    metrics['all']['hits5'] += 1
                    metrics[grp]['hits5'] += 1
                    metrics['all']['ndcg5'] += 1.0 / math.log2(top5_ids.index(true_id) + 2)
                    metrics[grp]['ndcg5'] += 1.0 / math.log2(top5_ids.index(true_id) + 2)

                if true_id in top10_ids:
                    metrics['all']['hits10'] += 1
                    metrics[grp]['hits10'] += 1
                    metrics['all']['ndcg10'] += 1.0 / math.log2(top10_ids.index(true_id) + 2)
                    metrics[grp]['ndcg10'] += 1.0 / math.log2(top10_ids.index(true_id) + 2)

                if true_id in top20_ids:
                    metrics['all']['hits20'] += 1
                    metrics[grp]['hits20'] += 1
                    metrics['all']['ndcg20'] += 1.0 / math.log2(top20_ids.index(true_id) + 2)
                    metrics[grp]['ndcg20'] += 1.0 / math.log2(top20_ids.index(true_id) + 2)

    pure_inference_duration = time.time() - inference_start_time
    if metrics['all']['count'] == 0:
        log.warning("No users were successfully evaluated.")
        return 0.0, 0.0, 0.0

    log.info(f"--- Inference Performance ---")
    log.info(f"  Total Users Evaluated: {metrics['all']['count']}")
    log.info(f"  Total Inference Time: {pure_inference_duration:.4f} seconds")
    log.info(f"  Inference Speed: {metrics['all']['count'] / max(pure_inference_duration, 1e-9):.2f} users/sec")

    log.info(f"--- Evaluation Summary (Overall) ---")
    for group_name, group_metrics in metrics.items():
        count = group_metrics['count']
        if count == 0:
            continue
        hit_rate = group_metrics['hits'] / count
        hit5_rate = group_metrics['hits5'] / count
        hit10_rate = group_metrics['hits10'] / count
        hit20_rate = group_metrics['hits20'] / count
        ndcg5_rate = group_metrics['ndcg5'] / count
        ndcg10_rate = group_metrics['ndcg10'] / count
        ndcg20_rate = group_metrics['ndcg20'] / count
        log.info(f"  --- Group: {group_name.upper()} ({count} items) ---")
        log.info(f"    Hit@1: {hit_rate:.6f}, Hit@5: {hit5_rate:.6f}, Hit@10: {hit10_rate:.6f}, Hit@20: {hit20_rate:.6f}")
        log.info(f"    NDCG@5: {ndcg5_rate:.6f}, NDCG@10: {ndcg10_rate:.6f}, NDCG@20: {ndcg20_rate:.6f}")
    log.info(f"---")

    return (
        metrics['all']['hits'] / metrics['all']['count'],
        metrics['all']['hits10'] / metrics['all']['count'],
        metrics['all']['ndcg10'] / metrics['all']['count']
    )
    
if __name__ == "__main__":
    args = _parse_args()

    log_file = args.log_file or './logs/eval/master_eval.log'
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    TQDM_FILE = open(log_file, 'w', buffering=1)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(TQDM_FILE)],
        force=True,
    )
    logger = logging.getLogger(__name__)
    pipeline_start_time = PROCESS_START_TIME
    logger.info("="*50 + "\nTiming Start (Process Boot)\n" + "="*50)
    logger.info(f"Process-start timestamp: {pipeline_start_time:.6f}")
    
    with open(args.config, 'r') as f:
        eval_config = _resolve_eval_data_paths(json.load(f))

    set_seed(
        eval_config.get('seed', 42),
        deterministic=eval_config.get('deterministic', True)
    )

    model, model_config = load_pretrained_model(eval_config['checkpoint'], device=eval_config['device'], logger=logger)
    if model_config is not None:
        model_config["eval_seed"] = int(eval_config.get("seed", 42))
    num_params_all = sum(p.numel() for p in model.parameters())
    logger.info(f"#params (total): {num_params_all:,}")

    preprocessing_start_time = time.time()
    logger.info("="*50 + "\nData Pre-processing\n" + "="*50)
    processed_data = process_data(
        seq_file=eval_config['seq_file'],
        cand_file=eval_config['cand_file'],
        dataset=eval_config['dataset'],
        logger=logger
    )
    if not processed_data:
        logger.error("Data processing failed. Exiting.")
        preprocessing_elapsed = time.time() - preprocessing_start_time
        logger.info(
            f"[Preprocessing] start -> failure exit: {preprocessing_elapsed:.4f}s "
            f"({_format_duration(preprocessing_elapsed)})"
        )
        total_elapsed = time.time() - pipeline_start_time
        logger.info(f"[E2E] Process start -> failure exit: {total_elapsed:.4f}s ({_format_duration(total_elapsed)})")
        exit()

    base_interaction_matrix = processed_data["interaction_matrix"]
    user_sequences = processed_data["eval_sequences"]
    item_popularity = processed_data["item_popularity"]
    predefined_candidates = processed_data["candidates"]

    logger.info("="*50 + "\nMode: GENERATE DATA\n" + "="*50)
    eval_mode = eval_config.get('mode', 'cand')
    if eval_mode not in ("cand", "full"):
        raise ValueError(f"Unsupported eval_config.mode: {eval_mode}")
    logger.info(f"Eval mode : {eval_mode}")

    num_users, num_items = base_interaction_matrix.shape

    generated_batches = generate_data(
        model,
        model_config=model_config,
        user_sequences=user_sequences,
        num_users=num_users,
        num_items=num_items,
        device=eval_config['device'],
        batch_size=eval_config.get('batch_size', 32),
        max_eval_users=eval_config.get('max_eval_users', None),
        support_set_size=eval_config.get('support_set_size', 4),
        predefined_candidates=predefined_candidates,
        mode=eval_mode,
        logger=logger,
        progress_file=TQDM_FILE,
    )

    logger.info("="*50 + "\nLoading Popularity Data and Defining Groups\n" + "="*50)

    sorted_items = sorted(item_popularity.items(), key=lambda item: item[1], reverse=True)
    num_items = len(sorted_items)

    head_boundary, torso_boundary = int(num_items * 0.2), int(num_items * 0.8)

    item_popularity_groups = {
        'head': {item_id for item_id, pop in sorted_items[:head_boundary]},
        'torso': {item_id for item_id, pop in sorted_items[head_boundary:torso_boundary]},
        'tail': {item_id for item_id, pop in sorted_items[torso_boundary:]}
    }
    logger.info(f"Item popularity groups created: Head={len(item_popularity_groups['head'])}, "
                f"Torso={len(item_popularity_groups['torso'])}, Tail={len(item_popularity_groups['tail'])}")

    preprocessing_elapsed = time.time() - preprocessing_start_time
    logger.info("="*50 + "\nPreprocessing Timing\n" + "="*50)
    logger.info(
        f"[Preprocessing] Data preprocessing + feature generation end: {preprocessing_elapsed:.4f}s "
        f"({_format_duration(preprocessing_elapsed)})"
    )

    inference_phase_start_time = time.time()
    logger.info("="*50 + "\nMode: INFERENCE\n" + "="*50)

    logger.info("="*50 + "\nMode: INFERENCE (OVERALL ALL USERS)\n" + "="*50)
    if model and item_popularity_groups is not None:
        inference(
            model=model,
            model_config=model_config,
            all_batches=generated_batches["overall"],
            item_popularity_groups=item_popularity_groups,
            device=eval_config['device'],
            data_tag="OVERALL",
            mode=eval_mode,
            logger=logger,
            progress_file=TQDM_FILE,
        )

    inference_phase_elapsed = time.time() - inference_phase_start_time
    logger.info("="*50 + "\nInference Phase Timing\n" + "="*50)
    logger.info(
        f"[Inference] inference phase total: {inference_phase_elapsed:.4f}s "
        f"({_format_duration(inference_phase_elapsed)})"
    )

    total_elapsed = time.time() - pipeline_start_time
    logger.info("="*50 + "\nEnd-to-End Timing\n" + "="*50)
    logger.info(f"[E2E] Process start -> evaluation end: {total_elapsed:.4f}s ({_format_duration(total_elapsed)})")
