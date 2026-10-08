# AR-WAM

**AR-WAM: A Visual-Conditioned Agent-Ready World Action Model for Robotic Manipulation**

[🌐 Project Website](https://ar-wam.github.io/) · [📑 PDF](https://arxiv.org/pdf/2609.23578) · [🗂️ Dataset](https://modelscope.cn/datasets/Jasonggg15/AR-WAM-RoboTwin-Dataset)

AR-WAM replaces language instructions with a **visual grounding prompt** denoting the interaction object and location, and a **learnable operation token** dictating the atomic skill to execute. It uses a frozen pretrained visual encoder and no language encoder, predicts scene evolution within compact latent states while decoding actions, and exposes explicit reasoning outputs for supervision.

## Method

![AR-WAM overview](https://ar-wam.github.io/image/overview.png)

A **reasoning expert** and a **world action expert** share a Transformer backbone. The reasoning expert predicts a target mask, terminal end-effector poses and a structured task abstract. The world action expert jointly learns actions and **dual-level future alignment**: visual targets come from an offline foresight-gist extractor, and semantic targets come from the shared reasoning backbone on future observations. Task conditioning combines atomic-operation embeddings and visual prompts through AdaLN.

See [Section III of the paper](https://arxiv.org/html/2609.23578v2#S3) for the method and the [project website](https://ar-wam.github.io/) for demonstrations and results.

## Installation

The reference environment uses Python 3.10, PyTorch 2.7.1, CUDA 12.8 and Transformers 5.0.0rc0.

```bash
conda env create -f environment.yml
conda activate ar-wam
```

Alternatively:

```bash
python -m pip install -r requirements.txt
```

Run all commands from the repository root. The examples use 8 GPUs; the GPU count must divide the configured global batch size of 128.

## Data and Pretrained Assets

Download the [dataset](https://modelscope.cn/datasets/Jasonggg15/AR-WAM-RoboTwin-Dataset) and DINOv3 weights into the following layout:

```text
AR-WAM/
├── data/arwam_release/
│   ├── robotwin/
│   │   ├── clean/
│   │   └── random/
│   ├── schema.json
│   └── norm_stats.json
└── weights/dinov3-vitl16-pretrain-lvd1689m/
```

### 1. Download the dataset

Download all task archives and metadata from ModelScope:

```bash
python -m pip install modelscope
mkdir -p data/arwam_release
modelscope download --dataset Jasonggg15/AR-WAM-RoboTwin-Dataset \
  --local_dir data/arwam_release
```

### 2. Verify and extract

Verify the downloaded ZIPs against `checksums.json`, then extract them (Linux, with `sha256sum` and `unzip` installed):

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('data/arwam_release')
entries = json.loads((root / 'checksums.json').read_text())
assert len(entries) == 100, 'Expected the complete 100-archive release'
with (root / 'SHA256SUMS').open('w') as output:
    for entry in entries:
        path = root / entry['path']
        assert path.stat().st_size == entry['bytes'], path
        output.write(f"{entry['sha256']}  {entry['path']}\n")
PY
(
  set -e
  cd data/arwam_release
  sha256sum -c SHA256SUMS
  for archive in clean/*.zip random/*.zip; do
    unzip -q -n "$archive" -d .
  done
)
```

Allow at least **1.1 TiB** of disk space for the archives (455 GiB) and extracted data (616 GiB). Each ZIP already contains `robotwin/<split>/<task>/...`; extract directly into `data/arwam_release/`.

### 3. Download the visual encoder

Download [DINOv3 ViT-L/16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) after obtaining access on Hugging Face:

```bash
hf download facebook/dinov3-vitl16-pretrain-lvd1689m \
  --local-dir weights/dinov3-vitl16-pretrain-lvd1689m
```

The configurations use these relative paths by default. For custom locations, update the paths in all three configuration files. Dataset fields and vocabularies are defined in `schema.json`; both policy stages use the provided `norm_stats.json`.

## Foresight-Gist Pretraining

Pretrain the offline foresight-gist extractor before policy training:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 train_rla.py \
  --config configs/rla_robotwin.yaml \
  --output-dir runs/rla
```

The configuration trains for 150,000 iterations and exports encoder/decoder weights and resumable checkpoints under `runs/rla/ckpts/`. Both policy configurations load the resulting encoder from `runs/rla`.

To resume:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 train_rla.py \
  --config configs/rla_robotwin.yaml \
  --output-dir runs/rla \
  --resume runs/rla/ckpts/latest.pt
```

## Policy Training

Stage 1 trains from scratch for 550,000 updates. Stage 2 loads Stage 1 model weights and trains for another 200,000 updates with a fresh optimizer and scheduler. Hyperparameters are provided in the configuration files.

```bash
# Stage 1
python -m torch.distributed.run --standalone --nproc_per_node=8 train.py \
  --config configs/stage1_robotwin_adaln.yaml \
  --norm-stats data/arwam_release/norm_stats.json \
  --output-dir runs/stage1

# Stage 2
python -m torch.distributed.run --standalone --nproc_per_node=8 train.py \
  --config configs/stage2_robotwin_adaln.yaml \
  --norm-stats data/arwam_release/norm_stats.json \
  --output-dir runs/stage2 \
  --warm-start runs/stage1/checkpoints/step_0550000.pt
```

Both policy stages save numbered checkpoints every 50,000 updates and refresh `latest.pt` every 10,000 updates, with final checkpoints saved at the stopping update.

To resume an interrupted policy run:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 train.py \
  --config configs/stage2_robotwin_adaln.yaml \
  --norm-stats data/arwam_release/norm_stats.json \
  --output-dir runs/stage2 \
  --resume runs/stage2/checkpoints/latest.pt
```

Use `--warm-start` for a new policy stage and `--resume` to restore training state. Resume requires matching training settings and assets; it does not restore RNG state or the data cursor. Both trainers require the same GPU count for resume; policy resume also requires the same normalization statistics. `--max-steps N` sets an absolute stopping iteration/update for short runs.

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{jiang2026arwam,
  title         = {{AR-WAM}: A Visual-Conditioned Agent-Ready World Action Model for Robotic Manipulation},
  author        = {Yicheng Jiang and Zesen Gan and Xiaobo Wang and Tianlun He and Chenxu Zhao and Minghui Wu and Xinyue Wang and Jiaxu Wang and Junhao He and Jianan Wang and Qiming Shao},
  year          = {2026},
  eprint        = {2609.23578},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2609.23578}
}
```

## Acknowledgments

Our code is built upon and adapted from [LiLa-WAM](https://github.com/teee000/LiLa-WAM) and [RLA-WM](https://github.com/mlzxy/rla-wm). We thank both projects for their open-source contributions.

If you are a rights holder and believe any included material infringes your rights, please contact us through a repository issue. We will promptly review the concern and remove the affected material where appropriate.

## License

Original AR-WAM contributions are released under the [MIT License](LICENSE). Inherited components and pretrained assets remain subject to their respective upstream licenses and notices.
