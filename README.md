# Neural Sequence Decoder

PyTorch implementation of a Neural Sequence Decoder for Speech BCI (Brain-Computer Interface). This project decodes neural signals into text sequences using a GRU-based architecture with Connectionist Temporal Classification (CTC) loss.

**Original Implementation**: [fwillett/speechBCI](https://github.com/fwillett/speechBCI/tree/main/NeuralDecoder)

## Features

This implementation includes several enhancements for improved model performance:

### 1. **Label Smoothing**
- **Fixed Label Smoothing**: Apply a constant smoothing factor (e.g., 0.05, 0.1) to prevent overconfidence
- **Progressive Label Smoothing**: Adaptive smoothing that decays from 0.15 to 0.05 over training, providing stronger regularization early and finer tuning later

### 2. **Layer Normalization**
- **Pre-norm**: Normalize inputs before the GRU layer
- **Post-norm**: Normalize outputs after the GRU layer (default)
- **Dual-norm**: Normalize both after input transformation and after GRU output

### 3. **Post-GRU Stack**
- Stack of Linear → LayerNorm → Dropout layers after the GRU
- Configurable number of layers and dropout rate
- Based on benchmark recommendations for improved sequence modeling

## Requirements

- Python >= 3.9
- PyTorch (see installation notes below)
- CUDA-capable GPU (recommended)

## Installation

### Standard Installation

```bash
pip install -e .
```

### For Google Colab (Python 3.9 + PyTorch 2.2.2)

If you're using Google Colab with Python 3.9 and need PyTorch 2.2.2:

```bash
# Install PyTorch first
pip install "torch==2.2.2" --index-url https://download.pytorch.org/whl/cu121

# Then install the package without dependencies
pip install -e . --no-deps --no-build-isolation
```

## Quick Start

### 1. Prepare Dataset

Convert the speech BCI dataset using the provided notebook:
```bash
jupyter notebook notebooks/formatCompetitionData.ipynb
```

### 2. Train a Single Model

```bash
python scripts/train_model.py
```

Edit `scripts/train_model.py` to customize training parameters.

### 3. Run Experiments

Run all experiments (baseline + architecture improvements):

```bash
python scripts/run_experiments.py
```

This will run 13 experiments:
- **Previous experiments**: baseline, label_smooth_0.05, label_smooth_0.1, layer_norm, combined_0.1_norm
- **New architecture experiments**: post_gru_stack, progressive_smoothing, layer_norm_pre, layer_norm_dual, progressive_post_gru_stack, pre_norm_post_gru_stack, dual_norm_post_gru_stack, progressive_dual_norm_post_gru_stack

### 4. Plot Results

Visualize and compare experiment results:

```bash
python scripts/plot_results.py
```

This will automatically detect and plot all completed experiments, showing:
- Test Loss over training
- Test CER (Character Error Rate) over training
- Comparison table with best/final metrics

## Configuration

### Model Architecture Parameters

Key parameters in `src/neural_decoder/conf/config.yaml`:

```yaml
# Label Smoothing (0.0 = disabled, typical: 0.05-0.1)
label_smoothing: 0.0

# Layer Normalization
use_layer_norm: False
layer_norm_position: post  # Options: "pre", "post", "dual", "none"

# Post-GRU Stack
use_post_gru_stack: False
post_gru_stack_layers: 2
post_gru_stack_dropout: 0.1
```

### Training Parameters

Default training configuration:
- **Batch Size**: 64
- **Learning Rate**: 0.02 (start) → 0.02 (end)
- **Hidden Units**: 1024
- **Layers**: 5
- **Bidirectional**: True
- **Training Batches**: 10,000
- **Evaluation Interval**: Every 100 batches

## Experiment Details

### Previous Experiments

1. **baseline**: No enhancements
2. **label_smooth_0.05**: Fixed label smoothing (0.05)
3. **label_smooth_0.1**: Fixed label smoothing (0.1)
4. **layer_norm**: Post-norm layer normalization
5. **combined_0.1_norm**: Label smoothing (0.1) + post-norm

### New Architecture Experiments

1. **post_gru_stack**: Post-GRU stack (2 layers, dropout=0.1)
2. **progressive_smoothing**: Progressive label smoothing (0.15 → 0.05)
3. **layer_norm_pre**: Pre-norm layer normalization
4. **layer_norm_dual**: Dual-norm layer normalization
5. **progressive_post_gru_stack**: Progressive smoothing + post-GRU stack
6. **pre_norm_post_gru_stack**: Pre-norm + post-GRU stack
7. **dual_norm_post_gru_stack**: Dual-norm + post-GRU stack
8. **progressive_dual_norm_post_gru_stack**: Progressive smoothing + dual-norm + post-GRU stack

## Project Structure

```
neural_seq_decoder/
├── src/
│   └── neural_decoder/
│       ├── model.py              # GRUDecoder architecture
│       ├── neural_decoder_trainer.py  # Training loop
│       ├── dataset.py            # Dataset loading
│       ├── augmentations.py      # Data augmentations
│       └── conf/
│           └── config.yaml       # Default configuration
├── scripts/
│   ├── train_model.py           # Single model training
│   ├── run_experiments.py       # Batch experiment runner
│   └── plot_results.py          # Results visualization
├── notebooks/
│   └── formatCompetitionData.ipynb  # Dataset preparation
└── README.md
```

## Output Format

Each experiment creates a directory with:
- `trainingStats`: Pickle file containing `testLoss` and `testCER` arrays
- `args`: Pickle file with experiment configuration
- Model checkpoints (if saved)

## Customization

### Adding a New Experiment

Edit `scripts/run_experiments.py`:

```python
{
    "name": "my_experiment",
    "label_smoothing": 0.1,
    "use_layer_norm": True,
    "layer_norm_position": "post",
    "use_post_gru_stack": True,
    "post_gru_stack_layers": 2,
    "post_gru_stack_dropout": 0.1
}
```

### Custom Training Configuration

Modify `scripts/train_model.py` or pass a custom `args` dictionary to `trainModel()`.

## Citation

If you use this code, please cite the original work:

```bibtex
@article{willett2023high,
  title={A high-performance speech neuroprosthesis},
  author={Willett, Francis R and Avansino, Donald T and Hochberg, Leigh R and Henderson, Jaimie M and Shenoy, Krishna V},
  journal={Nature},
  year={2023}
}
```

## License

MIT License (see LICENSE.txt)

## Authors

- Original: Chaofei Fan, Frank Willett (Stanford)
- Enhancements: [Your Name/Institution]

## Acknowledgments

Based on the original implementation by [fwillett/speechBCI](https://github.com/fwillett/speechBCI/tree/main/NeuralDecoder).
