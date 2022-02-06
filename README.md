<div align="center">

# [CVPR 2021 Challenge](https://www.aicitychallenge.org/)  <img src="assets/CVPR.png" alt="CVPR 2021" width="35" height="22">

## Track 5: Natural Language-Based Vehicle Retrieval

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## Overview

- This is the code for the #4 solution [[paper]](https://openaccess.thecvf.com/content/CVPR2021W/AICity/html/Nguyen_Contrastive_Learning_for_Natural_Language-Based_Vehicle_Retrieval_CVPRW_2021_paper.html)


<div align="center">
<img src="assets/leaderboard.png" alt="Track 4 private leader board" width="600" height="350">
</div>

## Prerequisites

- torch
- torchvision
- numpy
- pandas
- sklearn
- opencv-python
- efficientnet_pytorch
- transformers
- albumentations
- tqdm
- timm
- textdistance
- openai/CLIP

Running `setup.sh` also installs the dependencies

## Reproceduring

- Running [run_nce.py](src/run_nce.py) with the default arguments
## Citation
```
@inproceedings{nguyen2021contrastive,
  title={Contrastive learning for natural language-based vehicle retrieval},
  author={Nguyen, Tam Minh and Pham, Quang Huu and Doan, Linh Bao and Trinh, Hoang Viet and Nguyen, Viet-Anh and Phan, Viet-Hoang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4245--4252},
  year={2021}
}
```
