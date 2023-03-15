# https://academictorrents.com/details/7208a86910cc518ae8feaa9021bf7f8565b97644

export MODEL_NAME="stabilityai/stable-diffusion-2"
export dataset_name="../../../data/dataset/"

# accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --resume_from_checkpoint=latest \
#   --dataset_name=$dataset_name \
#   --use_ema \
#   --resolution=768 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --gradient_checkpointing \
#   --max_train_steps=15000 \
#   --learning_rate=1e-05 \ 
#   --max_grad_norm=1 \
#   --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --output_dir="ped-xray-model"

  accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --resume_from_checkpoint=latest \
  --dataset_name=$dataset_name --caption_column="text" \
  --resolution=768 \
  --train_batch_size=2 \
  --num_train_epochs=150 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="ped-xray-model-lora" \
  --validation_prompt="bacteria" --report_to="wandb"
