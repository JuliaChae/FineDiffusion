export HF_DATASETS_CACHE="/local/scratch/carlyn.1/hf_dataset_cache"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="/home/carlyn.1/code/FineDiffusion/pose_conditioning/my_datasets/ap_10k/ap_10k_no_cond.py"
export ACC_PORT=29503

accelerate launch --mixed_precision="fp16" --multi_gpu --main_process_port $ACC_PORT --num_processes=2  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=30000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/local/scratch/carlyn.1/diffusion_models/sd-ap-10k-model" \
  --resume_from_checkpoint "latest"