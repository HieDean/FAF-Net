# Reference-Based Speech Enhancement via Feature Alignment and Fusion Network

### Introduction

This repo provides a PyTorch implementation of the paper: [Reference-Based Speech Enhancement via Feature Alignment and Fusion Network](https://ojs.aaai.org/index.php/AAAI/article/view/21419).

![](https://github.com/HieDean/FAF-Net/blob/main/figure/framework.png)

### Requirements
pytorch>=1.10.0

torchaudio>=0.10.0

numpy>=1.21.0

tqdm>=4.64.0

pesq>=0.0.3

### Usage

1. Replace the vbd dataset path in ```dataset.py``` and ```ref_prepare.py``` .
2. ```python ref_prepare.py``` to generate 15s reference clips.
3. ```python test_stage_1.py``` to get the result of one stage model.
4. ```python test_stage_2.py``` to get the result of two stage model.

### Citation

If you find the code useful in your research, please cite:

```
@inproceedings{yue2022reference,
  title={Reference-based speech enhancement via feature alignment and fusion network},
  author={Yue, Huanjing and Duo, Wenxin and Peng, Xiulian and Yang, Jingyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={10},
  pages={11648--11656},
  year={2022}
}
```
