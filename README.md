# PREVISION: Fine-tuning LLaVA for Autonomous Driving Corner Case Analysis

<p align="center">
  <b>PRe-training Enhanced Versatile Integration of Semantics, Images, and Object Detection for Novel Corner Case Analysis in Autonomous Driving</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#results">Results</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#training">Training</a> •
  <a href="#inference">Inference</a>
</p>

---

## Overview

This project extends [LLaVA (Large Language and Vision Assistant)](https://github.com/haotian-liu/LLaVA) with a custom multimodal architecture for the **ECCV 2024 Autonomous Driving Challenge**. Our approach integrates:

- **RGB Images** via CLIP vision encoder
- **Object Detection** (34 autonomous driving classes) via custom bounding box encoder
- **Depth Information** via Depth-Anything-V2

The model performs three tasks:
1. **General Perception**: Describe all objects affecting the ego vehicle's driving behavior
2. **Regional Perception**: Explain specific objects highlighted in the scene
3. **Driving Suggestions**: Provide actionable driving recommendations based on scene understanding

## Results

| Algorithm | Settings | Final Score | BLEU |
|-----------|----------|:-----------:|:----:|
| LLaVA | Zero Shot (init prompt) | 2.41 | 0.18 |
| LLaVA | Zero Shot (revised prompt) | 3.03 | 0.32 |
| LLaVA | Finetune | × | × |
| LLaVA* | Finetune (LoRA) | 3.90 | 0.43 |
| LLaVA* | Finetune + Postprocess | **4.09** | **0.48** |
| Pretrained LLaVA* | Finetune (LoRA) | 3.77 | 0.43 |
| Pretrained LLaVA* | Finetune (DoRA) | 3.85 | 0.46 |
| Pretrained LLaVA* | Finetune + Multistage | ≤ 2 | × |

> **LLaVA\***: Our extended architecture with custom bounding box encoder

**Key Findings:**
- LoRA fine-tuning with post-processing achieved the best performance (**4.09**)
- Pre-training both projectors degraded performance due to error propagation from noisy detection labels
- Multi-stage inference (knowledge transfer) did not improve results in our setting

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Input Processing                               │
├─────────────────┬─────────────────────────┬─────────────────────────────┤
│   RGB Image     │   Bounding Box Maps     │      Depth Map              │
│   (3 channels)  │   (34 channels)         │      (1 channel)            │
└────────┬────────┴───────────┬─────────────┴──────────────┬──────────────┘
         │                    │                            │
         ▼                    └────────────┬───────────────┘
┌─────────────────┐                        ▼
│   CLIP ViT-L    │           ┌─────────────────────────────┐
│   (Frozen)      │           │   Custom BBox Encoder       │
│                 │           │   - Dual-stream processing  │
│                 │           │   - 4-layer Transformer     │
│                 │           │   - Positional embeddings   │
└────────┬────────┘           └──────────────┬──────────────┘
         │                                   │
         ▼                                   ▼
┌─────────────────┐           ┌─────────────────────────────┐
│  MM Projector   │           │      BBox Projector         │
│  (MLP 2x GELU)  │           │      (MLP 2x GELU)          │
└────────┬────────┘           └──────────────┬──────────────┘
         │                                   │
         └──────────────┬────────────────────┘
                        ▼
              ┌─────────────────┐
              │   Concatenate   │
              │   (576 + 196)   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   LLaMA-7B      │
              │   (LoRA r=4)    │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Text Output   │
              └─────────────────┘
```

### Custom Bounding Box Encoder

The bounding box encoder is a CLIP-inspired Transformer that processes object detection and depth information:

```python
# Input: 35 channels (34 object classes + 1 depth)
# Output: (batch_size, 196 patches, 4096 dimensions)

CLIPBoundingBoxEncoder(
    input_channels=35,      # 34 bbox classes + 1 depth
    hidden_dim=512,
    output_dim=4096,        # Match LLaMA hidden size
    patch_size=24,
    num_layers=4,
    num_heads=8,
    image_size=336
)
```

### Supported Object Classes (34 Categories)

```
Vehicles: car, truck, bus, motorcycle, bicycle, van, SUV, trailer, 
          moped, ambulance, construction vehicle

Road Users: pedestrian, cyclist, motorcyclists, road users

Traffic Control: red traffic light, traffic light, parking sign, 
                 warning traffic sign, directional traffic sign, 
                 traffic box, sentry box, traffic cone, traffic island, 
                 barrier, bollard

Miscellaneous: debris, machinery, dustbin, concrete block, dog, 
               chair, phone booth, streetlights
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/Unabashed0501/Finetune-LLAVA-on-Autonomous-Driving/
cd Finetune-LLAVA-on-Autonomous-Driving

# Create conda environment
conda create -n prevision python=3.10 -y
conda activate prevision

# Install dependencies
cd LLaVA
pip install -e .
pip install -r llava_requirements.txt

# Install Flash Attention (optional, for faster training)
pip install flash-attn --no-build-isolation

# Install additional dependencies
pip install deepspeed wandb ultralytics
```

### Model Weights

Download the base LLaVA model:
```bash
# The model will be automatically downloaded from HuggingFace
# Base model: liuhaotian/llava-v1.5-7b
```

## Usage

### Data Preparation

#### 1. Generate Depth Maps and Bounding Boxes

```bash
cd generate_pretrain_data
python gen_pretrain.py \
    --image_path /path/to/images \
    --output_depth_npy /path/to/depth_npy \
    --output_json /path/to/output.json
```

#### 2. Data Format

The training data should be in JSON format:

```json
[
    {
        "id": "train_general_001",
        "image": "/path/to/image.jpg",
        "depth_npy": "/path/to/depth.npy",
        "bounding_box": [
            {
                "category_name": "car",
                "bbox": [x_min, y_min, x_max, y_max]
            }
        ],
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nDescribe the objects in this driving scene."
            },
            {
                "from": "gpt",
                "value": "In this image, I can see..."
            }
        ]
    }
]
```

## Training

### Stage 1: Pre-training (Optional)

Pre-train the bounding box encoder on synthetic data:

```bash
bash pretrain.sh
```

Or run directly:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path /path/to/pretrain_data.json \
    --val_data_path /path/to/val_data.json \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --bb_projector_type mlp2x_gelu \
    --bb_input_dim 35 \
    --tune_bbox_encoder True \
    --freeze_mm_mlp_adapter True \
    --bb_encoder_lr 5e-4 \
    --bf16 True \
    --output_dir ./checkpoints/pretrain \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_steps 2000 \
    --learning_rate 1e-3 \
    --model_max_length 2048
```

### Stage 2: Fine-tuning with LoRA

Fine-tune on the target task:

```bash
bash train.sh
```

Or run directly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
deepspeed llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 4 \
    --lora_alpha 64 \
    --use_dora False \
    --mm_projector_lr 5e-5 \
    --bb_encoder_lr 5e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path /path/to/train.json \
    --val_data_path /path/to/val.json \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --bb_projector_type mlp2x_gelu \
    --bb_input_dim 35 \
    --tune_bbox_encoder False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ./checkpoints/finetune-lora \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --max_steps 6000 \
    --learning_rate 1e-5 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --report_to wandb
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--lora_enable` | Enable LoRA fine-tuning | `False` |
| `--lora_r` | LoRA rank | `64` |
| `--lora_alpha` | LoRA alpha | `16` |
| `--use_dora` | Use DoRA instead of LoRA | `False` |
| `--tune_bbox_encoder` | Train only bbox encoder (pre-training) | `False` |
| `--freeze_mm_mlp_adapter` | Freeze the MM projector | `False` |
| `--bb_input_dim` | Bbox encoder input channels | `35` |
| `--mm_projector_lr` | Learning rate for MM projector | `None` |
| `--bb_encoder_lr` | Learning rate for bbox encoder | `None` |
| `--pretrain_bbox_encoder` | Path to pretrained bbox encoder | `None` |

## Inference

### Single GPU Inference

```bash
python LLaVA/llava/eval/gen_car_output.py \
    --model-path ./checkpoints/finetune-lora \
    --model-base liuhaotian/llava-v1.5-7b \
    --image-folder /path/to/images \
    --annotation_file /path/to/test.json \
    --output_file submission.json \
    --temperature 0.2 \
    --max_new_tokens 512
```

### Multi-GPU Parallel Inference

For faster inference on large test sets:

```bash
python parallel_inference.py
```

This will:
1. Split the test dataset across available GPUs
2. Run inference in parallel
3. Merge results into a single submission file

### Inference Shell Script

```bash
bash inference.sh
```

## Project Structure

```
Finetune-LLAVA-on-Autonomous-Driving/
├── LLaVA/
│   ├── llava/
│   │   ├── model/
│   │   │   ├── llava_arch.py           # Extended LLaVA architecture
│   │   │   ├── boundingbox_encoder/    # Custom bbox encoder
│   │   │   │   ├── boundingbox_encoder.py
│   │   │   │   └── builder.py
│   │   │   └── language_model/
│   │   │       └── llava_llama.py
│   │   ├── train/
│   │   │   ├── train.py                # Main training script
│   │   │   ├── llava_trainer.py        # Custom trainer
│   │   │   └── llama_flash_attn_monkey_patch.py
│   │   ├── eval/
│   │   │   └── gen_car_output.py       # Inference script
│   │   └── constants.py                # Object classes definition
│   └── scripts/
│       └── v1_5/
│           ├── pretrain.sh
│           └── finetune_task_lora.sh
├── generate_pretrain_data/
│   ├── gen_pretrain.py                 # Generate pre-training data
│   └── gen_conversation.py             # Generate QA pairs
├── utils/
│   ├── depth_anything.py
│   └── dino.py
├── parallel_inference.py               # Multi-GPU inference
├── train.sh
├── pretrain.sh
├── inference.sh
└── README.md
```

## Ablation Studies

### Effect of Pre-training

| Pre-training | Fine-tuning | Score |
|--------------|-------------|:-----:|
| ❌ | LoRA | **3.90** |
| ✅ (bbox only) | LoRA | 3.77 |
| ✅ (bbox + mm_proj) | LoRA | 3.65 |

**Conclusion**: Pre-training degraded performance due to noisy labels from YOLOv8-World.

### Effect of LoRA Configuration

| LoRA Rank | Alpha | Score |
|:---------:|:-----:|:-----:|
| 4 | 64 | **3.90** |
| 8 | 64 | 3.85 |
| 16 | 128 | 3.78 |

### Effect of Learning Rates

| LLM LR | Projector LR | Score |
|:------:|:------------:|:-----:|
| 1e-5 | 5e-5 | **3.90** |
| 2e-5 | 5e-5 | 3.82 |
| 1e-5 | 1e-4 | 3.75 |

## Tips & Troubleshooting

### Out of Memory (OOM)

- Reduce `per_device_train_batch_size`
- Enable `gradient_checkpointing`
- Use DeepSpeed ZeRO-3 (modify `--deepspeed ./scripts/zero3.json`)

### Slow Training

- Enable Flash Attention: Install `flash-attn` package
- Use bf16 training: `--bf16 True`
- Increase `dataloader_num_workers`

### Poor Performance

- Ensure depth maps are normalized to [0, 1]
- Check bounding box coordinates are in correct format (x_min, y_min, x_max, y_max)
- Verify image preprocessing matches CLIP's expected format

## Citation

If you use this code, please cite:

```bibtex
@misc{prevision2024,
  title={PREVISION: Fine-tuning LLaVA for Autonomous Driving Corner Case Analysis},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/Finetune-LLAVA-on-Autonomous-Driving}}
}
```

## Acknowledgements

- [LLaVA](https://github.com/haotian-liu/LLaVA) - Base multimodal model
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) - Depth estimation
- [YOLOv8-World](https://github.com/ultralytics/ultralytics) - Open-vocabulary object detection
- [CODA-LM](https://coda-dataset.github.io/) - Dataset and challenge

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
