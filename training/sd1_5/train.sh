#!/bin/bash
# Launch script for InstructPix2Pix training on 8xA100 GPUs
# Uses the official HuggingFace Diffusers training script

echo "=========================================="
echo "InstructPix2Pix Training (SD 1.5)"
echo "Using Official HuggingFace Script"
echo "=========================================="

# Default values
DATA_DIR="./data"
OUTPUT_DIR="./output/instruct-pix2pix-sd15"
MODEL_NAME="runwayml/stable-diffusion-v1-5"
RESOLUTION=512
TRAIN_BATCH_SIZE=8
NUM_EPOCHS=100
LEARNING_RATE=5e-5
CHECKPOINTING_STEPS=1000
VALIDATION_STEPS=1000
SEED=42
RESUME_FROM_CHECKPOINT=""
ORIGINAL_IMAGE_COLUMN="input_image"
EDIT_PROMPT_COLUMN="edit_prompt"
EDITED_IMAGE_COLUMN="edited_image"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --train_batch_size)
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --checkpointing_steps)
            CHECKPOINTING_STEPS="$2"
            shift 2
            ;;
        --resume_from_checkpoint)
            RESUME_FROM_CHECKPOINT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Build the training command
CMD="accelerate launch --mixed_precision=fp16 --multi_gpu --num_processes=8 train_instruct_pix2pix.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATA_DIR \
  --original_image_column=$ORIGINAL_IMAGE_COLUMN \
  --edit_prompt_column=$EDIT_PROMPT_COLUMN \
  --edited_image_column=$EDITED_IMAGE_COLUMN \
  --resolution=$RESOLUTION \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --num_train_epochs=$NUM_EPOCHS \
  --learning_rate=$LEARNING_RATE \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_grad_norm=1.0 \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --conditioning_dropout_prob=0.05 \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --validation_steps=$VALIDATION_STEPS \
  --output_dir=$OUTPUT_DIR \
  --seed=$SEED \
  --enable_xformers_memory_efficient_attention \
  --report_to=wandb"

# Add resume checkpoint if provided
if [ ! -z "$RESUME_FROM_CHECKPOINT" ]; then
    CMD="$CMD --resume_from_checkpoint=$RESUME_FROM_CHECKPOINT"
fi

# Display configuration
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "  Batch size per GPU: $TRAIN_BATCH_SIZE"
echo "  Global batch size: $((TRAIN_BATCH_SIZE * 8))"
echo "  Number of epochs: $NUM_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  GPUs: 8"
echo "  Mixed precision: fp16"
echo "  Seed: $SEED"
if [ ! -z "$RESUME_FROM_CHECKPOINT" ]; then
    echo "  Resume from: $RESUME_FROM_CHECKPOINT"
fi
echo ""
echo "Starting training..."
echo "=========================================="
echo ""

# Run training
eval $CMD
