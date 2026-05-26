export CUDA_VISIBLE_DEVICES=0
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1

nohup python train.py \
  --config train_config.json \
  --model_name srpfn > /dev/null 2>&1 &
