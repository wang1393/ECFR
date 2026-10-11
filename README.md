# ECFR: An Efficient Cross-modal Feature Reconstruction Model for Multimodal Multi-class Anomaly Detection

**Zhenghan Wang · Zhihui Luo · Haitao Hu · Wengang Cheng**

Official implementation of **An Efficient Cross-modal Feature Reconstruction Model for Multimodal Multi-class Anomaly Detection**, accepted at **NeurIPS 2026**.

## Contents

- [Overview](#overview)
- [Closely Related Work](#closely-related-work)
  - [What Distinguishes ECFR?](#what-distinguishes-ecfr)
  - [BibTeX for Related Work](#bibtex-for-related-work)
- [Installation](#installation)
- [Datasets](#datasets)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Overview

**Efficient Cross-modal Feature Reconstruction (ECFR)** detects and localizes anomalies across multiple object categories using a single model trained on normal RGB–depth pairs. A shared, frozen ResNet-34 extracts multi-scale features, and two decoding pathways reconstruct RGB features from depth and depth features from RGB. Cross-modal reconstruction exploits inter-modal heterogeneity to alleviate the “identical shortcut” problem, in which anomalous features can also be well reconstructed.

ECFR has two core modules:

- **Adaptive Feature Interaction and Recalibration (AFIR)** models intra- and inter-modal dependencies and adaptively recalibrates feature channels.
- **Hybrid Attention Convolution (HAC)** combines [Agent Attention](https://github.com/LeapLabTHU/Agent-Attention) with parallel multi-kernel depth-wise convolutions to model global dependencies and local structures during reconstruction.

At inference time, discrepancies between reconstructed and observed features are used for anomaly detection and localization.

![Overall framework of ECFR](figures/ecfr_framework.png)

## Closely Related Work

We acknowledge the following closely related works, grouped by the number of object categories handled by one model and the input modalities used. Single-class methods train a separate model for each object category; multi-class methods handle multiple categories with one model.

| Setting | Related work | Main approach |
| --- | --- | --- |
| Multi-class, single-modal | [UniAD](https://github.com/zhiyuanyou/UniAD) — NeurIPS 2022; [MambaAD](https://github.com/lewandofskee/MambaAD) — NeurIPS 2024 | Unified feature reconstruction using query-based decoding or state space models |
| Single-class, multimodal | [CFM](https://github.com/CVLAB-Unibo/crossmodal-feature-mapping) — CVPR 2024; [M3DM](https://github.com/wynoxos/M3DM) — CVPR 2023; [G2SF](https://github.com/ctaoaa/G2SF) — ICCV 2025 | Cross-modal feature mapping or multimodal memory-based detection and fusion |
| Multi-class, multimodal — contemporary work | [UniMMAD](https://github.com/yuanzhao-CVLAB/UniMMAD) — CVPR 2026 | MoE-driven feature decompression |

The grouping reflects the original research settings. Single-class methods can also be adapted and evaluated under a unified multi-class protocol.

### What Distinguishes ECFR?

The works above explore unified feature reconstruction, multimodal memory banks, cross-modal feature mapping, and recent unified multimodal approaches. Among them, [CFM](https://arxiv.org/html/2312.04521v2) maps contextualized features from frozen encoders independently at each spatial location. **During bidirectional cross-modal feature reconstruction, ECFR combines cross-modal interaction and multi-scale supervision with global and local modeling to learn to reconstruct spatial structures.** This design connects cross-modal correspondence learning ([CFM](https://arxiv.org/html/2312.04521v2)) with spatially structured reconstruction for multi-class anomaly detection, while helping to alleviate the “identical shortcut” problem.

### BibTeX for Related Work

<details>
<summary>Show BibTeX entries</summary>

```bibtex
@inproceedings{you2022uniad,
  title={A Unified Model for Multi-class Anomaly Detection},
  author={You, Zhiyuan and Cui, Lei and Shen, Yujun and Yang, Kai and Lu, Xin and Zheng, Yu and Le, Xinyi},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}

@inproceedings{he2024mambaad,
  title={MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection},
  author={He, Haoyang and Bai, Yuhu and Zhang, Jiangning and He, Qingdong and Chen, Hongxu and Gan, Zhenye and Wang, Chengjie and Li, Xiangtai and Tian, Guanzhong and Xie, Lei},
  booktitle={Advances in Neural Information Processing Systems},
  volume={37},
  year={2024}
}

@inproceedings{costanzino2024cfm,
  title={Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping},
  author={Costanzino, Alex and Zama Ramirez, Pierluigi and Lisanti, Giuseppe and Di Stefano, Luigi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17234--17243},
  year={2024}
}

@inproceedings{wang2023m3dm,
  title={Multimodal Industrial Anomaly Detection via Hybrid Fusion},
  author={Wang, Yue and Peng, Jinlong and Zhang, Jiangning and Yi, Ran and Wang, Yabiao and Wang, Chengjie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8032--8041},
  year={2023}
}

@inproceedings{tao2025g2sf,
  title={G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection},
  author={Tao, Chengyu and Cao, Xuanming and Du, Juan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20551--20560},
  year={2025}
}

@inproceedings{zhao2026unimmad,
  title={UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression},
  author={Zhao, Yuan and Pang, Youwei and Zhang, Lihe and Liu, Hanqi and Zuo, Jiaming and Lu, Huchuan and Zhao, Xiaoqi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}

```

</details>

## Installation

The experiments in the paper use PyTorch 1.13. Install a compatible CUDA-enabled PyTorch and torchvision pair, then install the remaining dependencies:

```bash
git clone https://github.com/wang1393/ECFR.git
cd ECFR
pip install -r requirements.txt
```

## Datasets

Download [MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) and [Eyecandies](https://eyecan-ai.github.io/eyecandies/) and organize the prepared data as follows:

```text
data/
├── mvtec3d/
│   ├── meta.json
│   └── ...
└── eyecandies_preprocessed/
    ├── meta.json
    └── ...
```

## Training and Evaluation

Run the commands from the repository root. Each configuration trains one model across all categories in its dataset.

```bash
python run.py -c configs/mvtec3d.py -m train
python run.py -c configs/eyecandies.py -m train
```

Evaluate with trained checkpoints:

```bash
python run.py -c configs/mvtec3d.py -m test \
  model.kwargs.checkpoint_path=/path/to/ecfr_mvtec3d.pth

python run.py -c configs/eyecandies.py -m test \
  model.kwargs.checkpoint_path=/path/to/ecfr_eyecandies.pth
```

Dataset paths can be changed with `data.root=/path/to/mvtec3d` or `data.root=/path/to/eyecandies_preprocessed`.

## Results

Multi-class results reported in the paper (all metrics in %):

| Dataset | I-AUROC | P-AUROC | AUPRO |
| --- | ---: | ---: | ---: |
| MVTec 3D-AD | 91.85 | 98.60 | 95.37 |
| Eyecandies | 86.64 | 97.13 | 90.21 |

The model has **44.627M parameters** and requires **15.762 GFLOPs**; the paper reports **21.356 FPS** on an NVIDIA A800 80GB GPU.

![Qualitative anomaly localization on MVTec 3D-AD](figures/ecfr_qualitative.png)

## Citation

Official BibTeX: pending publication.

## Acknowledgement

We thank the authors of the related works above for their contributions to anomaly detection research.

We sincerely thank the NeurIPS 2026 reviewers and area chairs (ACs) for their constructive feedback, and the senior area chairs (SACs) and program chairs (PCs) for their efforts in overseeing the review process.

We appreciate the following github repos for their valuable code:

- [UniAD](https://github.com/zhiyuanyou/UniAD)
- [MambaAD](https://github.com/lewandofskee/MambaAD)
- [CFM](https://github.com/CVLAB-Unibo/crossmodal-feature-mapping)
- [ADer](https://github.com/zhangzjn/ADer#train-multi-class-unsupervised-ad-setting-by-default-muad)
