import argparse
import copy
import json
import logging
import os
import time

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformer import SRPFN
from srpfn.dataset import RecIterableDataset
from utils import (
    ema_update_,
    get_cosine_schedule_with_warmup,
    get_ema_decay,
    get_encoder_generator,
)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu_device', default='cuda', type=str)
    parser.add_argument('--model_name', type=str, default='srpfn')
    args = parser.parse_args()
    return args

def train(args):
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        return None, None
    
    log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    model_name = args.model_name
    log_file_path = os.path.join(log_dir, f"train_{model_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file_path)]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*50)
    logger.info(f"Loaded Configuration: {json.dumps(config, indent=2)}")
    logger.info("="*50)

    training_params = config['train']
    grad_accum_steps = training_params.get('grad_accum_steps', 1)
    model_params = config['model']
    data_params = config['data']
    env_params = config['environment']
    
    num_available_gpus = torch.cuda.device_count()
    if num_available_gpus >= 2:
        train_gpu_idx = env_params.get('train_gpu_idx', 0)
        train_device = torch.device(f"cuda:{train_gpu_idx}")
        logger.info("Multi-GPU mode enabled.")
    elif num_available_gpus == 1:
        device_str = args.gpu_device if args.gpu_device.startswith('cuda') else 'cuda:0'
        train_device = torch.device(device_str)
        logger.info("Single-GPU mode enabled.")
    else:
        train_device = torch.device("cpu")
        logger.info("No CUDA devices found. Running on CPU.")
    
    logger.info("Setting up Dataset and DataLoader...")
    try:    
        dataset = RecIterableDataset(
            max_support_set_size=data_params['max_support_set_size'],
            num_candidates=data_params['num_candidates'],
            feature_dim=data_params['feature_dim'],
            batch_size=training_params['batch_size'],
            steps_per_epoch=training_params['steps_per_epoch'],
            max_history_len=data_params['max_history_len'],
            device='cpu',
        )

        num_workers = training_params.get('num_workers', 4)
        dataloader_kwargs = dict(
            dataset=dataset,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        if num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = 2
            dataloader_kwargs["persistent_workers"] = True

        dataloader = DataLoader(**dataloader_kwargs)

        feature_dim = data_params['feature_dim']
        emsize = model_params['emsize']
        encoder_generator = get_encoder_generator()

        model = SRPFN(
            feature_dim=feature_dim,
            emsize=emsize,
            interaction_encoder=encoder_generator,
            nhead=model_params['nhead'],
            nhid=model_params['nhid_factor'] * emsize,
            nlayers=model_params['nlayers'],
            max_history_len=data_params['max_history_len'],
            dropout=model_params['dropout'],
            activation=model_params['activation'],
            input_normalization=model_params['input_normalization'],
        )
    except Exception as e:
        logger.error("Failed to create dataset/model.", exc_info=True)
        raise
    model.to(train_device)
    
    ema_model = copy.deepcopy(model).to(train_device)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_params.get('lr', 0.0001), weight_decay=training_params['weight_decay'])

    steps_per_epoch_effective = training_params['steps_per_epoch'] // grad_accum_steps
    num_warmup_steps = training_params['warmup_epochs'] * steps_per_epoch_effective
    num_training_steps = training_params['epochs'] * steps_per_epoch_effective
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        eta_min_ratio=0.01
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(training_params['use_amp'] and train_device.type == 'cuda'))

    logger.info(f"Starting training from epoch 1 to {training_params['epochs']}...")
    total_steps_taken = 0
    
    for epoch in range(1, training_params['epochs'] + 1):
        epoch_start_time = time.time()
        model.train()
        total_loss_sum = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", total=training_params['steps_per_epoch'], ncols=100)

        try:
            for batch_idx, batch in enumerate(pbar):
                if batch_idx >= training_params['steps_per_epoch']:
                    break

                interaction_vectors = batch["interaction_vectors"].to(train_device, non_blocking=True)

                position_ids = batch["position_ids"].to(train_device, non_blocking=True)
                labels = batch["labels"].to(train_device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(train_device, non_blocking=True) 

                answer_positions = batch["answer_positions"].to(train_device, non_blocking=True)
                
                candidate_counts = batch["candidate_counts"].to(train_device, non_blocking=True)
                candidate_interaction_vecs = batch["candidate_interaction_vecs"].to(train_device, non_blocking=True)
                
                support_interaction_vecs = batch["support_interaction_vecs"].to(train_device, non_blocking=True)
                support_valid_mask = batch["support_valid_mask"].to(train_device, non_blocking=True)
                
                del batch
                
                if interaction_vectors.shape[1] == 0:
                    continue
                
                if (batch_idx % grad_accum_steps == 0):
                    optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=(training_params['use_amp'] and train_device.type=='cuda')):
                    output = model(
                        interaction_vectors=interaction_vectors,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        answer_positions=answer_positions,
                        
                        support_interaction_vecs=support_interaction_vecs,
                        support_valid_mask=support_valid_mask,

                        candidate_interaction_vecs=candidate_interaction_vecs,
                        candidate_counts=candidate_counts,
                        return_logits=True,
                    )
                    task_loss = criterion(output["logits"], labels)
                    
                    loss = task_loss
                    loss = loss / grad_accum_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                    scaler.step(optimizer)
                    scaler.update()

                    total_steps_taken += 1
                    decay = get_ema_decay(total_steps_taken, base_decay=0.999, warmup_steps=num_warmup_steps)
                    ema_update_(ema_model, model, decay)

                    scheduler.step()

                total_loss_sum += loss.item() * grad_accum_steps
                current_loss_val = loss.item() * grad_accum_steps

                if (batch_idx + 1) % grad_accum_steps == 0:
                    pbar.set_postfix({
                        'Task Loss': f"{current_loss_val:.4f}", 
                        'LR': f"{optimizer.param_groups[0]['lr']:.6f}"
                    })

                log_interval = 10 * grad_accum_steps 
                if (batch_idx + 1) % log_interval == 0:
                    logger.info(
                        f"Epoch {epoch} | Step {batch_idx+1}/{training_params['steps_per_epoch']} | "
                        f"Loss: {current_loss_val:.4f} | "
                        f"LR: {optimizer.param_groups[0]['lr']:.8f}"
                    )
                    
        except (Exception, KeyboardInterrupt) as e:
            logger.error(f"Error during training epoch {epoch}: {e}", exc_info=True)
            raise

        num_steps_in_epoch = batch_idx + 1
        avg_epoch_loss = total_loss_sum / num_steps_in_epoch if num_steps_in_epoch > 0 else 0
        epoch_duration_wall_clock = time.time() - epoch_start_time

        logger.info(f"--- Epoch {epoch} Summary ---")
        logger.info(f"  Wall Clock Time: {epoch_duration_wall_clock:.2f}s")
        logger.info(f"  Average Loss: {avg_epoch_loss:.4f}")
        logger.info(f"---")

        checkpoint_dir = env_params.get('save_path', './models')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_{model_name}.pt')

        model_to_save = model
        save_data = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'ema_model_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'total_steps_taken': total_steps_taken,
        }

        try:
            torch.save(save_data, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint to {checkpoint_path}: {e}", exc_info=True)
                        
    logger.info("Training finished.")
    model_to_return = model
    return model_to_return.to('cpu'), config
    
if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    args = _parse_args()
    train(args)
