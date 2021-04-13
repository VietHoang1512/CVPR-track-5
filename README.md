<div align="center">

# [CVPR 2021 Challenge](https://www.aicitychallenge.org/)

## Track 5: Natural Language-Based Vehicle Retrieval

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## Overview
- This is the code for EmoRecCom #5 solution
- The [single model](https://drive.google.com/file/d/1-2-rGO46ZXVS0GTxxx-LlyVe_RU3aumu/view?usp=sharing) achieved 0.1564 MRR in the private leaderboard

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