# GMT: A Geometric Multigrid Transformer Solver for Microstructure Homogenization

Official implementation of **GMT: A Geometric Multigrid Transformer Solver for Microstructure Homogenization**, accepted to the **SIGGRAPH 2026 Journal Track**.


<!-- [![ACM](https://img.shields.io/static/v1?label=ACM&message=10.1145/3811333&color=blue&logo=acm)](https://dl.acm.org/doi/abs/10.1145/3637528.3671961) &emsp;&emsp;  -->
[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=2604.26518&color=red&logo=arxiv)](https://arxiv.org/abs/2604.26518)

![GMT overview](assets/teaser.jpg)

GMT is a neural solver for large-scale microstructure homogenization. It combines sparse 3D feature extraction, Point Transformer V3 blocks, and a geometric multigrid solver to accelerate the linear elasticity solves that dominate high-resolution homogenization pipelines.

This repository provides a pure-Jittor implementation:

- Training and automatic differentiation based on Jittor
- Sparse voxel preprocessing utilities
- Serialized sparse point attention without spconv or flash-attn
- Matrix-free element-by-element geometric multigrid correction
- The default experiment configuration used by the release
- Checkpoint and TensorBoard logging support

## Repository Layout

```text
.
|-- configs/train.yaml          # Default training configuration
|-- datagen/generate.py         # Raw voxel to sparse .npz preprocessing
|-- environment.yml             # Conda environment
|-- main.py                     # Training entry point
`-- train/
    |-- dataset.py              # Sparse dataset and collate function
    |-- model.py                # GMT model
    |-- EBE_GMG.py              # Element-by-element geometric multigrid solver
    |-- PTv3_3.py               # Point Transformer V3 backbone
    |-- sp_lightning.py         # Jittor training loop (legacy filename)
    `-- _utils.py               # FEM assembly and solver utilities
```

## Environment

The Jittor environment is captured in `environment.yml`.
See the [Jittor homepage](https://cg.cs.tsinghua.edu.cn/jittor/) and
[Jittor API documentation](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html).

Requirements:

- Linux with an NVIDIA GPU is recommended
- Conda or Mamba

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate GMT-jittor
```

Jittor compiles operators for the current machine on first use. The migrated code
does not depend on PyTorch, Lightning, spconv, flash-attn, or torch-scatter.

The Jittor sparse point backbone is API-equivalent at the training boundary, but
its operator implementation differs from the original spconv/flash-attn model.
Existing PyTorch checkpoints are therefore not compatible and the model must be
retrained.

## Data Format

Training expects processed `.npz` files. Each file should contain:

- `coords`: active node coordinates, shape `(N, 3)`
- `node_type`: per-node local occupancy features, shape `(N, 8)`
- `voxel`: binary solid voxel grid, shape `(R, R, R)`
- `node_index`: element-to-node connectivity, shape `(E, 8)`

<!-- A typical processed-data layout is:

```text
data/Train/PSL
data/Train/Truss
data/Train/TPMS
data/Vail/PSL
data/Vail/Truss
data/Vail/TPMS
```

Update `train_data_path` and `val_data_path` in `configs/train.yaml` to match your local dataset paths. The default release config uses `resolution: 64`. -->

## Preprocessing Voxels

Use `datagen/generate.py` to convert raw voxel grids into the processed `.npz` format.

Supported raw inputs:

- `.npy`: dense voxel array
- `.csv`: flattened or dense voxel values
- `.npz`: must contain a `voxel` array

Example:

```bash
python datagen/generate.py \
  --input_dirs path/to/raw_voxels \
  --out_dir data/Train/Truss \
  --res 64 \
  --type Truss
```

The script treats positive values as solid voxels and writes one processed `.npz` file per input sample.

## Training

Edit `configs/train.yaml` before training. The most commonly changed fields are:

- `train_data_path`, `val_data_path`: processed training and validation folders
- `batch_size`: sparse batch size
- `learning_rate`, `warmup_steps`, `gradient_clip_norm`: optimizer stability settings
- `resolution`: voxel resolution
- `model.*`: transformer depth, channels, heads, and window sizes
- `GMG.*`: geometric multigrid smoothing and cycle settings
- `pre_train`: checkpoint path for resuming training, or `null`

Run on one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py configs/train.yaml
```

Run with Jittor MPI:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 mpirun --bind-to none -np 3 \
  python main.py configs/train.yaml
```

`batch_size` is the per-GPU batch size. With three MPI processes and
`batch_size: 3`, the effective global batch size is 9. Jittor assigns one MPI
process to each visible GPU and averages gradients through MPI/NCCL.

Set `device: cpu` in the YAML config for CPU execution. Jittor uses CUDA when
`device: gpu` and a supported CUDA installation are available.

## Outputs

Outputs are written under `output_path` in the config. With the default config:

```text
result/
|-- checkpoint/     # Jittor model state dictionaries
`-- tf_logs/        # TensorBoard logs
```

Launch TensorBoard with:

```bash
tensorboard --logdir result/tf_logs
```

Run the one-step model smoke test in a Linux Jittor environment:

```bash
PYTHONPATH=. python tests/smoke_test.py
CUDA_VISIBLE_DEVICES=0 JITTOR_USE_CUDA=1 PYTHONPATH=. python tests/smoke_test.py
```
<!-- 
## Current Release Notes

- This snapshot focuses on training and voxel preprocessing.
- Public dataset download links and pretrained checkpoints are not included in this repository snapshot.
- If you use a different voxel resolution, update both the preprocessing `--res` argument and `resolution` in the config.

## Citation

If you use this code, please cite:

```bibtex
@misc{xing2026gmt,
  title        = {GMT: A Geometric Multigrid Transformer Solver for Microstructure Homogenization},
  author       = {Xing, Yu and Liu, Yang and Xue, Tianyang and Lu, Lin},
  year         = {2026},
  eprint       = {2604.26518},
  archivePrefix = {arXiv},
  note         = {Accepted to the SIGGRAPH 2026 Journal Track}
}
``` -->
