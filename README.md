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

A major challenge in popular reconstruction networks is the “identical shortcut” problem, where anomalous samples can also be well reconstructed, hindering the discrimination between normal and abnormal data. This challenge is further exacerbated in multi-class settings, as category diversity compels the model to develop stronger reconstruction capacity to capture more complex normal patterns, which may inadvertently allow abnormal samples to be well reconstructed as well. Meanwhile, compared to single-modal reconstruction, cross-modal reconstruction is more challenging due to the inter-modal heterogeneity and typically yields larger reconstruction errors. In this research, we argue **the difficulty of cross-modal reconstruction is not an impediment, but a blessing in disguise**: it naturally suppresses the identical shortcut and enhances anomaly detectability. Specifically, we establish two statistical inequalities that formalize our core hypotheses: cross-modal heterogeneity inherently mitigates the identical shortcut (Inequality 1) and amplifies anomaly detectability (Inequality 2).

**Inequality 1 (Cross-modal Difficulty)**

$$
\mathbb{E}_{\mathcal{P}_{\mathrm{norm}}}
\left[
\left\lVert
\mathcal{F}_{\mathrm{cross}}(\mathbf{x}^{\bar m})-\mathbf{x}^{m}
\right\rVert_{2}^{2}
\right]
\geq
\mathbb{E}_{\mathcal{P}_{\mathrm{norm}}}
\left[
\left\lVert
\mathcal{F}_{\mathrm{single}}(\mathbf{x}^{m})-\mathbf{x}^{m}
\right\rVert_{2}^{2}
\right]
+\Delta_{\mathrm{hetero}},
$$

**Inequality 2 (Anomaly Amplification)**

$$
\mathbb{E}_{\mathcal{P}_{\mathrm{anom}}}
\left[
\left\lVert
\mathcal{F}_{\mathrm{cross}}(\mathbf{x}^{m})-\mathbf{x}^{\bar m}
\right\rVert_{2}^{2}
\right]
\geq
\kappa\cdot
\mathbb{E}_{\mathcal{P}_{\mathrm{norm}}}
\left[
\left\lVert
\mathcal{F}_{\mathrm{cross}}(\mathbf{x}^{m})-\mathbf{x}^{\bar m}
\right\rVert_{2}^{2}
\right].
$$

![Overall framework of ECFR](figures/ecfr_framework.png)


## Closely Related Work

We appreciate the contributions of the authors of the closely related studies listed below. The studies are grouped by their original training settings: single-class methods train a separate model for each category, while multi-class methods use one model across categories.

## <sub>Multi-class, single-modal</sub>

<details>
<summary>You et al. (2022), <strong>A Unified Model for Multi-class Anomaly Detection (UniAD)</strong> (<a href="https://proceedings.neurips.cc/paper_files/paper/2022/hash/1d774c112926348c3e25ea47d87c835b-Abstract-Conference.html">NeurIPS</a>, <a href="https://github.com/zhiyuanyou/UniAD">code</a>). </summary>

```bibtex
@inproceedings{you2022unified,
  title = {A Unified Model for Multi-class Anomaly Detection},
  author = {You, Zhiyuan and Cui, Lei and Shen, Yujun and Yang, Kai and Lu, Xin and Zheng, Yu and Le, Xinyi},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {35},
  pages = {4571--4584},
  year = {2022}
}
```

</details>

UniAD pioneered unified multi-class feature reconstruction and highlighted the *identical shortcut* problem.

<details>
<summary>He et al. (2024), <strong>MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection</strong> (<a href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/833b21da1956c6b92f6df253bf655cf5-Abstract-Conference.html">NeurIPS</a>, <a href="https://github.com/lewandofskee/MambaAD">code</a>). </summary>

```bibtex
@inproceedings{he2024mambaad,
  title = {MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection},
  author = {He, Haoyang and Bai, Yuhu and Zhang, Jiangning and He, Qingdong and Chen, Hongxu and Gan, Zhenye and Wang, Chengjie and Li, Xiangtai and Tian, Guanzhong and Xie, Lei},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  pages = {71162--71187},
  year = {2024}
}
```

</details>

MambaAD uses state-space scans and multi-kernel convolutions to model global and local structure in single-modal feature reconstruction. ECFR also models spatial structure, and uses cross-modal reconstruction to exploit RGB–depth heterogeneity against the *identical shortcut*.

## <sub>Single-class, multimodal</sub>

<details>
<summary>Wang et al. (2023), <strong>Multimodal Industrial Anomaly Detection via Hybrid Fusion (M3DM)</strong> (<a href="https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Multimodal_Industrial_Anomaly_Detection_via_Hybrid_Fusion_CVPR_2023_paper.html">CVPR</a>, <a href="https://github.com/wynoxos/M3DM">code</a>). </summary>


```bibtex
@inproceedings{wang2023multimodal,
  title = {Multimodal Industrial Anomaly Detection via Hybrid Fusion},
  author = {Wang, Yue and Peng, Jinlong and Zhang, Jiangning and Yi, Ran and Wang, Yabiao and Wang, Chengjie},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {8032--8041},
  year = {2023}
}
```

</details>

M3DM fuses RGB and 3D features and stores normal features in multiple memory banks. ECFR learns cross-modal feature reconstruction without a memory bank.

<details>
<summary>Costanzino et al. (2024), <strong>Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping (CFM)</strong> (<a href="https://openaccess.thecvf.com/content/CVPR2024/html/Costanzino_Multimodal_Industrial_Anomaly_Detection_by_Crossmodal_Feature_Mapping_CVPR_2024_paper.html">CVPR</a>, <a href="https://github.com/CVLAB-Unibo/crossmodal-feature-mapping">code</a>). </summary>

```bibtex
@inproceedings{costanzino2024cross,
  title = {Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping},
  author = {Costanzino, Alex and Zama Ramirez, Pierluigi and Lisanti, Giuseppe and Di Stefano, Luigi},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {17234--17243},
  year = {2024}
}
```

</details>

CFM maps contextualized features independently at each spatial location. ECFR learns spatial structure through global and local modeling during cross-modal feature reconstruction across categories.

<details>
<summary>Tao et al. (2025), <strong>G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection</strong> (<a href="https://openaccess.thecvf.com/content/ICCV2025/html/Tao_G2SF_Geometry-Guided_Score_Fusion_for_Multimodal_Industrial_Anomaly_Detection_ICCV_2025_paper.html">ICCV</a>, <a href="https://github.com/ctaoaa/G2SF">code</a>). </summary>

```bibtex
@inproceedings{tao2025g2sf,
  title = {G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection},
  author = {Tao, Chengyu and Cao, Xuanming and Du, Juan},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages = {20551--20560},
  year = {2025}
}
```

</details>

G2SF learns a geometry-guided metric to fuse modality-specific memory-bank scores. ECFR scores cross-modal reconstruction discrepancies directly.

## <sub>Multi-class, multimodal</sub>

<details>
<summary>Zhao et al. (2026), <strong>UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression</strong> (<a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_UniMMAD_Unified_Multi-Modal_and_Multi-Class_Anomaly_Detection_via_MoE-Driven_Feature_CVPR_2026_paper.pdf">CVPR</a>, <a href="https://github.com/yuanzhao-CVLAB/UniMMAD">code</a>). </summary>

```bibtex
@inproceedings{zhao2026unimmad,
  title = {UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression},
  author = {Zhao, Yuan and Pang, Youwei and Zhang, Lihe and Liu, Hanqi and Zuo, Jiaming and Lu, Huchuan and Zhao, Xiaoqi},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {28502--28511},
  year = {2026}
}
```

</details>

UniMMAD uses a mixture of experts to decompress shared features into modality- and class-specific representations. ECFR uses two interacting pathways for cross-modal feature reconstruction.

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

For questions, please send an email to [17325527132@163.com](mailto:17325527132@163.com).

## Acknowledgements

We sincerely thank all reviewers and the Area Chair for their constructive comments during the review of our submission. Their feedback helped strengthen this work.

We appreciate the following GitHub repositories for their valuable code: [UniAD](https://github.com/zhiyuanyou/UniAD), [MambaAD](https://github.com/lewandofskee/MambaAD), and [ADer](https://github.com/zhangzjn/ADer).
