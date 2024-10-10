export HF_DATASETS_CACHE="/local/scratch/carlyn.1/hf_dataset_cache"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export CUSTOM_UNET_PATH="/local/scratch/carlyn.1/diffusion_models/sd-ap-10k-model/"
export T2I_MODEL_NAME="TencentARC/t2iadapter_openpose_sd14v1"
#export T2I_MODEL_NAME="TencentARC/t2iadapter_keypose_sd14v1"
export DATASET_NAME="/home/carlyn.1/code/FineDiffusion/pose_conditioning/my_datasets/ap_10k/ap_10k.py"
export ACC_PORT=29505

# --pretrained_unet_path=$CUSTOM_UNET_PATH \
#--adapter_model_name_or_path=$T2I_MODEL_NAME \
#--adam_epsilon 0.001 \

accelerate launch --mixed_precision="fp16" --main_process_port $ACC_PORT --multi_gpu --num_processes=2 train_t2i_adapter_sd.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --adapter_model_name_or_path=$T2I_MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=1000 \
  --learning_rate=1e-6 \
  --adam_epsilon 0.001 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/local/scratch/carlyn.1/diffusion_models/sd-ap-10k-model-T2I-pose-PT-test" \
  --validation_image="/home/carlyn.1/code/FineDiffusion/pose_conditioning/notebooks/test_pose.png" \
  --validation_prompt="a photo of a squirrel" \
  --validation_steps=10 \
