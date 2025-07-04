# EVA-CLIP to CLIP Flow Matching for 3D Vision

A research project implementing flow matching to bridge EVA-CLIP and CLIP L-14 representation spaces for downstream 3D model applications.

## 🎯 Project Overview

This project develops a trainable pipeline that maps EVA-CLIP image representations to CLIP ViT-L/14 feature space using flow matching with Lumina-Next. The goal is to leverage EVA-CLIP's superior visual encoding while maintaining compatibility with existing 3D models built on CLIP L-14 features.

### Architecture

```
Image → EVA-CLIP L-14 → [Cross-Attention] → Lumina-Next DiT → CLIP ViT-L/14 Features
                                ↑
                              Noise → Flow Matching
```

## 🏗️ Technical Approach

- **Source Encoder**: EVA-CLIP L-14 for robust visual feature extraction
- **Target Space**: CLIP ViT-L/14 (768-dim) for 3D model compatibility  
- **Flow Matching**: Lumina-Next DiT architecture with cross-attention conditioning
- **Training**: Flow matching loss between predicted and ground truth CLIP features

## 📂 Project Structure

```
eva-clip-flow-matching/
├── src/
│   ├── data/                 # Dataset loading and preprocessing
│   ├── models/               # Model implementations
│   ├── training/             # Training pipeline
│   └── evaluation/           # Metrics and evaluation
├── config/                   # Configuration files
├── scripts/                  # Utility scripts
├── notebooks/                # Exploration and analysis
└── cache/                    # Cached features (not in git)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Access to Snellius computing cluster (for dataset storage)
- CUDA-capable GPU (recommended)

### Installation

```bash
git clone https://github.com/your-org/eva-clip-flow-matching.git
cd eva-clip-flow-matching
pip install -r requirements.txt
```

### Dataset Setup

We use the BLIP3o-pretrain-short-caption dataset. For initial development, download one shard:

```bash
# On Snellius
cd /path/to/your/data
python scripts/download_data.py --shard 0 --num_samples 1000
```

### Feature Caching

Pre-compute and cache EVA-CLIP and CLIP features for faster training:

```bash
python scripts/cache_features.py --dataset_path /path/to/data --shard 0
```

### Training

```bash
python src/training/train.py --config config/model_config.yaml
```

## 📊 Current Status

- ✅ **Phase 1**: Data pipeline and feature extraction setup
- 🔄 **Phase 2**: Lumina-Next implementation (in progress)
- ⏳ **Phase 3**: Training pipeline and evaluation
- ⏳ **Phase 4**: Integration with 3D model pipeline

## 🔧 Development Notes

### Model Specifications

- **EVA-CLIP L-14**: 768-dimensional image features
- **CLIP ViT-L/14**: 768-dimensional target features  
- **Lumina-Next**: DiT-based flow matching model
- **Flow Matching**: Continuous normalizing flows for representation mapping

### Key Design Decisions

1. **Feature Caching**: Pre-compute embeddings to accelerate training iterations
2. **Single Shard Training**: Start with 1K samples for rapid prototyping
3. **Cross-Attention**: Use EVA features to condition the flow matching process
4. **Modular Architecture**: Separate data, models, and training for easy experimentation

## 📖 References

- [EVA: Exploring the Limits of Masked Visual Representation Learning at Scale](https://arxiv.org/abs/2211.07636)
- [Lumina-Next: Making Lumina-T2X Stronger and Faster with Next-DiT](https://arxiv.org/abs/2406.18583)
- [BLIP-3: Building Large-scale Vision-Language Models](https://arxiv.org/abs/2408.11060)

## 👥 Team

- **Graduate Student**: [Mohammad Hafeez Khan] - Pipeline implementation 


## 🔬 Research Context

This work is part of a larger effort to build unified 3D models that can leverage multiple vision-language representations. By creating learnable mappings between different encoder spaces, we aim to combine the strengths of various models while maintaining compatibility with existing pipelines.

## 📝 License

[Add your institution's license here]

## 🤝 Contributing

This is an active research project. For questions or collaboration:
- Open an issue for bugs or feature requests
- Contact the team for research collaboration opportunities

---

**Note**: This project is in active development. Documentation and code will be updated frequently as we progress through the implementation phases.
