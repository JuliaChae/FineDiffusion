export CUDA_VISIBLE_DEVICES=1,2
export HF_DATASETS_CACHE="/local/scratch/carlyn.1/hf_dataset_cache"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATASET_NAME="/home/carlyn.1/code/FineDiffusion/pose_conditioning/my_datasets/ap_10k/ap_10k_no_cond.py"
export ACC_PORT=29507

accelerate launch --mixed_precision="fp16" --main_process_port $ACC_PORT --multi-gpu --num_processes=2  train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=30000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/local/scratch/carlyn.1/diffusion_models/sdxl-ap-10k-model" \