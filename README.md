# ECFR: An Efficient Cross-modal Feature Reconstruction Model for Multimodal Multi-class Anomaly Detection

**Zhenghan Wang · Zhihui Luo · Haitao Hu · Wengang Cheng**

Official implementation of **An Efficient Cross-modal Feature Reconstruction Model for Multimodal Multi-class Anomaly Detection**, accepted at **NeurIPS 2026**.

## Contents

- [Overview](#overview)
- [Closely Related Work](#closely-related-work)
- [Installation](#installation)
- [Datasets](#datasets)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Citation](#citation)
- [Contacts](#contacts)
- [Acknowledgements](#acknowledgements)

## Overview

In multi-class anomaly detection, a single model must learn normal patterns across multiple object categories. Yet greater reconstruction capacity can also allow anomalies to be reconstructed successfully. This is the *identical shortcut* problem. **Modality differences make cross-modal reconstruction more difficult. Can this difficulty help alleviate the identical shortcut?** Motivated by this question, ECFR explores how to exploit the inherent difficulty of cross-modal reconstruction to make anomalies harder to reconstruct while learning normal patterns across categories.

This idea rests on two hypotheses. First, modality heterogeneity introduces an additional cost when reconstructing one modality from another. Second, when the model learns this process from normal data alone, anomalous inputs may further increase reconstruction errors. We express these hypotheses through two inequalities and test them empirically in the appendix:

$$
\mathbb{E}_{P_{\mathrm{norm}}}
\left[\left\|F_{\mathrm{cross}}(x^{\bar m})-x^m\right\|_2^2\right]
\geq
\mathbb{E}_{P_{\mathrm{norm}}}
\left[\left\|F_{\mathrm{single}}(x^m)-x^m\right\|_2^2\right]
+\Delta_{\mathrm{hetero}},
$$

$$
\mathbb{E}_{P_{\mathrm{anom}}}
\left[\left\|F_{\mathrm{cross}}(x^m)-x^{\bar m}\right\|_2^2\right]
\geq
\kappa\,
\mathbb{E}_{P_{\mathrm{norm}}}
\left[\left\|F_{\mathrm{cross}}(x^m)-x^{\bar m}\right\|_2^2\right].
$$

Here, $x^m$ and $x^{\bar m}$ are paired inputs from two modalities. $F_{\mathrm{cross}}/F_{\mathrm{single}}$ denote cross-modal/same-modal reconstruction, with expectations taken over normal or anomalous data. $\Delta_{\mathrm{hetero}}>0$ captures the heterogeneity cost, and $\kappa>1$ characterizes the amplification of reconstruction errors on anomalous samples relative to normal samples.

As shown in the figure below, ECFR implements this idea through two interacting cross-modal reconstruction pathways. **AFIR** learns intra-modal and inter-modal dependencies from normal samples. These interactions provide a route for anomaly signals to propagate between the pathways. **HAC** jointly models global and local structure, allowing propagated deviations to produce more pronounced reconstruction errors.

![Overall framework of ECFR](figures/ecfr_framework.png)


## Closely Related Work

We appreciate the contributions of the authors of the closely related studies listed below. The studies are grouped by their original training settings: single-class methods train a separate model for each category, while multi-class methods use one model across categories.

**Multi-class, single-modal**

- You et al. (2022), **[A Unified Model for Multi-class Anomaly Detection (UniAD)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1d774c112926348c3e25ea47d87c835b-Abstract-Conference.html)** (NeurIPS, [code](https://github.com/zhiyuanyou/UniAD)). UniAD pioneered unified multi-class feature reconstruction and highlighted the *identical shortcut* problem.
- He et al. (2024), **[MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection](https://proceedings.neurips.cc/paper_files/paper/2024/hash/833b21da1956c6b92f6df253bf655cf5-Abstract-Conference.html)** (NeurIPS, [code](https://github.com/lewandofskee/MambaAD)). MambaAD uses state-space scans and multi-kernel convolutions to model global and local structure in single-modal feature reconstruction. ECFR also models spatial structure, and uses cross-modal reconstruction to exploit RGB–depth heterogeneity against the *identical shortcut*.

**Single-class, multimodal**

- Wang et al. (2023), **[Multimodal Industrial Anomaly Detection via Hybrid Fusion (M3DM)](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Multimodal_Industrial_Anomaly_Detection_via_Hybrid_Fusion_CVPR_2023_paper.html)** (CVPR, [code](https://github.com/wynoxos/M3DM)). M3DM fuses RGB and 3D features and stores normal features in multiple memory banks. ECFR learns cross-modal feature reconstruction without a memory bank.
- Costanzino et al. (2024), **[Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping (CFM)](https://openaccess.thecvf.com/content/CVPR2024/html/Costanzino_Multimodal_Industrial_Anomaly_Detection_by_Crossmodal_Feature_Mapping_CVPR_2024_paper.html)** (CVPR, [code](https://github.com/CVLAB-Unibo/crossmodal-feature-mapping)). CFM maps contextualized features independently at each spatial location. ECFR learns spatial structure through global and local modeling during cross-modal feature reconstruction across categories.
- Tao et al. (2025), **[G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection](https://openaccess.thecvf.com/content/ICCV2025/html/Tao_G2SF_Geometry-Guided_Score_Fusion_for_Multimodal_Industrial_Anomaly_Detection_ICCV_2025_paper.html)** (ICCV, [code](https://github.com/ctaoaa/G2SF)). G2SF learns a geometry-guided metric to fuse modality-specific memory-bank scores. ECFR scores cross-modal reconstruction discrepancies directly.

**Multi-class, multimodal**

- Zhao et al. (2026), **[UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_UniMMAD_Unified_Multi-Modal_and_Multi-Class_Anomaly_Detection_via_MoE-Driven_Feature_CVPR_2026_paper.pdf)** (CVPR, [code](https://github.com/yuanzhao-CVLAB/UniMMAD)). UniMMAD uses a mixture of experts to decompress shared features into modality- and class-specific representations. ECFR uses two interacting pathways for cross-modal feature reconstruction.

ECFR’s cross-modal feature reconstruction connects cross-modal correspondence learning with spatially structured reconstruction for multi-class anomaly detection. It further exploits modality heterogeneity to help mitigate the *identical shortcut*.

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

The model has **44.627M parameters** and requires **15.762 GFLOPs**. The paper reports **21.356 FPS** on an NVIDIA A800 80GB GPU.

![Qualitative anomaly localization on MVTec 3D-AD](figures/ecfr_qualitative.png)

## Citation

Official BibTeX: pending publication.

## Contacts

For questions, please send an email to [120242227180@ncepu.edu.cn](mailto:120242227180@ncepu.edu.cn).

## Acknowledgements

We thank all of the reviewers for their constructive comments and the Area Chair for helpful guidance during the review of our submission. Both helped us improve the paper.

We appreciate the following GitHub repositories for their valuable code:

- [UniAD](https://github.com/zhiyuanyou/UniAD)
- [MambaAD](https://github.com/lewandofskee/MambaAD)
- [ADer](https://github.com/zhangzjn/ADer#train-multi-class-unsupervised-ad-setting-by-default-muad)
