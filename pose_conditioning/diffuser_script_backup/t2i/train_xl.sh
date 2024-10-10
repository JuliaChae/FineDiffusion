export CUDA_VISIBLE_DEVICES=0,3
export HF_DATASETS_CACHE="/local/scratch/carlyn.1/hf_dataset_cache"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
#export CUSTOM_UNET_PATH="/local/scratch/carlyn.1/diffusion_models/sd-ap-10k-model/"
export T2I_MODEL_NAME="TencentARC/t2i-adapter-openpose-sdxl-1.0"
export DATASET_NAME="/home/carlyn.1/code/FineDiffusion/pose_conditioning/my_datasets/ap_10k/ap_10k.py"
#export DATASET_NAME="/home/carlyn.1/code/FineDiffusion/pose_conditioning/my_datasets/animalpose/animalpose.py"
export ACC_PORT=29506

# --pretrained_unet_path=$CUSTOM_UNET_PATH \
# --adapter_model_name_or_path=$T2I_MODEL_NAME \

accelerate launch --mixed_precision="fp16" --main_process_port $ACC_PORT --multi_gpu --num_processes=2 train_t2i_adapter_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=20000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/local/scratch/carlyn.1/diffusion_models/sdxl-animalpose-model-T2I-pose-solo-xl" \
  --validation_image="/home/carlyn.1/code/FineDiffusion/pose_conditioning/notebooks/test_animalpose.png" \
  --validation_prompt="a photo of a cat" \
  --validation_steps=10 \
  #--resume_from_checkpoint "latest" \
