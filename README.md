# Contour-enhanced-Visual-State-Space-Model
G-VMamba: Contour-enhanced Visual State-Space Model for Remote Sensing Image Classification

This repo is the PyTorch implementation of some works related to remote sensing tasks. A complete code implementation will be provided after the paper is accepted.

## Introduction
The current branch has been tested on Linux system, PyTorch 1.13.X and CUDA 11.6, supports Python 3.8+.

## Overview

* **Class activation map (CAM) visualization of the final normalization layer for [VMamba](https://github.com/MzeroMiko/VMamba) and G-VMamba Models’ classification of UC-Merced dataset images.**

When the model classifies the scenes in the image, the G-VMamba model focuses on areas where the color (or brightness) of the image changes more significantly (red areas), such as the lane intersection position of the Overpass scene, the edge of the court in the Baseball diamond scene, and the airplane shadow and lawn border of the Airplane scene. (The model size is ‘Small’.)
<p align="center">
  <img src="figures/CAM1.png" alt="arch" width="35%">
</p>


* **The overall architecture: (a) Overview of G-VMamba model; (b) Feature grouping in the G-VSS block.**

<p align="center">
  <img src="figures/overall_architecture.png" alt="arch" width="80%">
</p>


## Preparing the dataset

<details open>

### Remote Sensing Image Classification Dataset

We provide the method of preparing the remote sensing image classification dataset used in the paper.

#### UC Merced Dataset

- Image and annotation download link: [UC Merced Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html).

#### AID Dataset

- Image and annotation download link: [AID Dataset](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets)。

#### NWPU RESISC45 Dataset

- Image and annotation download link: [NWPU RESISC45 Dataset](https://aistudio.baidu.com/datasetdetail/220767)。

## Installation
**Step 1.** Create a conda environment and activate it.

```shell
conda create -n Gvmamba python=3.9
conda activate Gvmamba
```

**Step 2.** Install the VMamba.
Please refer to the [code support](https://github.com/MzeroMiko/VMamba).

**Step 3.** Install the requirements.
- Torch1.13.1 + cu116
```shell
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

**Step 3.** Complete installation will be provided later.
……
