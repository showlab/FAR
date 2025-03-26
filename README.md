# 🎥 FAR: Frame Autoregressive Model for Both Short- and Long-Context Video Modeling 🚀

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://farlongctx.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2503.19325-b31b1b.svg)](https://arxiv.org/abs/2503.19325)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-FAR-yellow)](https://huggingface.co/guyuchao/FAR_Models)&nbsp;


</div>


<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/abs/2503.19325">Long-Context Autoregressive Video Modeling with Next-Frame Prediction</a>
</p>

![dmlab_sample](./assets/dmlab_sample.png)

## 📢 News

* **2025-03:** Paper and Code of [FAR](https://farlongctx.github.io/) are released! 🎉


## 🌟 What's the Potential of FAR?

### 🔥 Introducing FAR: a new baseline for autoregressive video generation

FAR (i.e., <u>**F**</u>rame <u>**A**</u>utoregressive <u>**M**</u>odel) learns to predict continuous frames based on an autoregressive context. Its objective aligns well with video modeling, similar to the next-token prediction in language modeling.

![dmlab_sample](./assets/pipeline.png)

### 🔥 FAR achieves better convergence than video diffusion models with the same continuous latent space

<p align="center">
<img src="./assets/converenge.jpg" width=55%>
<p>

### 🔥 FAR leverages clean visual context without additional image-to-video fine-tuning:

Unconditional pretraining on UCF-101 achieves state-of-the-art results in both video generation (context frame = 0) and video prediction (context frame ≥ 1) within a single model.

<p align="center">
<img src="./assets/performance.png" width=75%>
<p>

### 🔥 FAR supports 16x longer temporal extrapolation at test time

<p align="center">
<img src="./assets/extrapolation.png" width=100%>
<p>

### 🔥 FAR supports efficient training on long-video sequence with managable token lengths

<p align="center">
<img src="./assets/long_short_term_ctx.jpg" width=55%>
<p>

#### 📚 For more details, check out our [paper](https://arxiv.org/abs/2503.19325).


## 🏋️‍♂️ FAR Model Zoo
We provide trained FAR models in our paper for re-implementation.

### Video Generation

We use seed-[0,2,4,6] in evaluation, following the evaluation prototype of [Latte](https://arxiv.org/abs/2401.03048):

| Model (Config) | #Params | Resolution | Condition | FVD | HF Weights | Pre-Computed Samples |
|:-------:|:------------:|:------------:|:-----------:|:-----:|:----------:|:----------:|
| [FAR-L](options/train/far/video_generation/FAR_L_ucf101_uncond_res128_400K_bs32.yml) | 457 M | 128x128 | ✗ | 280 ± 11.7 | [Model-HF](https://huggingface.co/guyuchao/FAR_Models/resolve/main/video_generation/FAR_L_UCF101_Uncond128-c19abd2c.pth) | [Google Drive](https://drive.google.com/drive/folders/1p1MvCiTfoUYAUYNqQNG6nEU02zy8U1vp?usp=drive_link) |
| [FAR-L](options/train/far/video_generation/FAR_L_ucf101_cond_res128_400K_bs32.yml) | 457 M | 128x128 | ✓ | 99 ± 5.9 | [Model-HF](https://huggingface.co/guyuchao/FAR_Models/resolve/main/video_generation/FAR_L_UCF101_Cond128-c6f798bf.pth) | [Google Drive](https://drive.google.com/drive/folders/1p1MvCiTfoUYAUYNqQNG6nEU02zy8U1vp?usp=drive_link) |
| [FAR-L](options/train/far/video_generation/FAR_L_ucf101_uncond_res256_400K_bs32.yml) | 457 M | 256x256 | ✗ | 303 ± 13.5 | [Model-HF](https://huggingface.co/guyuchao/FAR_Models/resolve/main/video_generation/FAR_L_UCF101_Uncond256-adea51e9.pth) | [Google Drive](https://drive.google.com/drive/folders/1p1MvCiTfoUYAUYNqQNG6nEU02zy8U1vp?usp=drive_link) |
| [FAR-L](options/train/far/video_generation/FAR_L_ucf101_cond_res256_400K_bs32.yml) | 457 M | 256x256 | ✓ | 113 ± 3.6 | [Model-HF](https://huggingface.co/guyuchao/FAR_Models/resolve/main/video_generation/FAR_L_UCF101_Cond256-41c6033f.pth) | [Google Drive](https://drive.google.com/drive/folders/1p1MvCiTfoUYAUYNqQNG6nEU02zy8U1vp?usp=drive_link) |
| [FAR-XL](options/train/far/video_generation/FAR_XL_ucf101_uncond_res256_400K_bs32.yml) | 657 M | 256x256 | ✗ | 279 ± 9.2 | [Model-HF](https://huggingface.co/guyuchao/FAR_Models/resolve/main/video_generation/FAR_XL_UCF101_Uncond256-3594ce6b.pth) | [Google Drive](https://drive.google.com/drive/folders/1p1MvCiTfoUYAUYNqQNG6nEU02zy8U1vp?usp=drive_link) |
| [FAR-XL](options/train/far/video_generation/FAR_XL_ucf101_cond_res256_400K_bs32.yml) | 657 M | 256x256 | ✓ | 108 ± 4.2 | [Model-HF](https://huggingface.co/guyuchao/FAR_Models/resolve/main/video_generation/FAR_XL_UCF101_Cond256-28a88f56.pth) | [Google Drive](https://drive.google.com/drive/folders/1p1MvCiTfoUYAUYNqQNG6nEU02zy8U1vp?usp=drive_link) |

###  Short-Video Prediction

We follows the evaluation prototype of [MCVD](https://arxiv.org/abs/2205.09853) and [ExtDM](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_ExtDM_Distribution_Extrapolation_Diffusion_Model_for_Video_Prediction_CVPR_2024_paper.pdf):

| Model (Config) | #Params | Dataset | PSNR | SSIM | LPIPS | FVD | HF Weights | Pre-Computed Samples |
|:-----:|:------------:|:------------:|:-----:|:-----:|:-----:|:-----:|:----------:|:----------:|
| [FAR-B](options/train/far/short_video_prediction/FAR_B_ucf101_res64_200K_bs32.yml) | 130 M | UCF101 | 25.64 | 0.818 | 0.037 | 194.1 | [Model-HF](https://huggingface.co/guyuchao/FAR_Models/resolve/main/short_video_prediction/FAR_B_UCF101_Uncond64-381d295f.pth) | [Google Drive](https://drive.google.com/drive/folders/1p1MvCiTfoUYAUYNqQNG6nEU02zy8U1vp?usp=drive_link) |
| [FAR-B](options/train/far/short_video_prediction/FAR_B_bair_res64_200K_bs32.yml) | 130 M | BAIR (c=2, p=28) | 19.40 | 0.819 | 0.049 | 144.3 | [Model-HF](https://huggingface.co/guyuchao/FAR_Models/resolve/main/short_video_prediction/FAR_B_BAIR_Uncond64-1983191b.pth) | [Google Drive](https://drive.google.com/drive/folders/1p1MvCiTfoUYAUYNqQNG6nEU02zy8U1vp?usp=drive_link) |

###  Long-Video Prediction

We use seed-[0,2,4,6] in evaluation, following the evaluation prototype of [TECO](https://arxiv.org/abs/2210.02396):


| Model (Config) | #Params | Dataset | PSNR | SSIM | LPIPS | FVD | HF Weights | Pre-Computed Samples |
|:-----:|:------------:|:------------:|:-----:|:-----:|:-----:|:-----:|:----------:|:----------:|
| [FAR-B-Long](options/train/far/long_video_prediction/FAR_B_Long_dmlab_res64_400K_bs32.yml) | 150 M | DMLab | 22.3 | 0.687 | 0.104 | 64 | [Model-HF](https://huggingface.co/guyuchao/FAR_Models/resolve/main/long_video_prediction/FAR_B_Long_DMLab_Action64-c09441dc.pth) | [Google Drive](https://drive.google.com/drive/folders/1p1MvCiTfoUYAUYNqQNG6nEU02zy8U1vp?usp=drive_link) |
| [FAR-M-Long](options/train/far/long_video_prediction/FAR_M_Long_minecraft_res128_400K_bs32.yml) | 280 M | Minecraft | 16.9 | 0.448 | 0.251 | 39 | [Model-HF](https://huggingface.co/guyuchao/FAR_Models/resolve/main/long_video_prediction/FAR_M_Long_Minecraft_Action128-4c041561.pth) | [Google Drive](https://drive.google.com/drive/folders/1p1MvCiTfoUYAUYNqQNG6nEU02zy8U1vp?usp=drive_link) |

## 🔧 Dependencies and Installation

### 1. Setup Environment:

```bash
# Setup Conda Environment
conda create -n FAR python=3.10
conda activate FAR

# Install Pytorch
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# Install Other Dependences
pip install -r requirements.txt
```

### 2. Prepare Dataset:

We have uploaded the dataset used in this paper to Hugging Face datasets for faster download. Please follow the instructions below to prepare.

```python
from huggingface_hub import snapshot_download, hf_hub_download

dataset_url = {
    "ucf101": "guyuchao/UCF101",
    "bair": "guyuchao/BAIR",
    "minecraft": "guyuchao/Minecraft",
    "minecraft_latent": "guyuchao/Minecraft_Latent",
    "dmlab": "guyuchao/DMLab",
    "dmlab_latent": "guyuchao/DMLab_Latent"
}

for key, url in dataset_url.items():
    snapshot_download(
        repo_id=url,
        repo_type="dataset",
        local_dir=f"datasets/{key}",
        token="input your hf token here"
    )
```

Then, enter its directory and execute:

```bash
find . -name "shard-*.tar" -exec tar -xvf {} \;
```


### 3. Prepare Pretrained Models of FAR:

We have uploaded the pretrained models of FAR to Hugging Face models. Please follow the instructions below to download if you want to evaluate FAR.

```bash
from huggingface_hub import snapshot_download, hf_hub_download

for key, url in dataset_url.items():
    snapshot_download(
        repo_id="guyuchao/FAR_Models",
        repo_type="model",
        local_dir="experiments/pretrained_models/FAR_Models",
        token="input your hf token here"
    )
```

## 🚀 Training

To train different models, you can run the following command:

```bash
accelerate launch \
    --num_processes 8 \
    --num_machines 1 \
    --main_process_port 19040 \
    train.py \
    -opt train_config.yml
```

* **Wandb:** Set ```use_wandb``` to ```True``` in config to enable wandb monitor.
* **Periodally Evaluation:** Set ```val_freq``` to control the peroidly evaluation in training.
* **Auto Resume:** Directly rerun the script, the model will find the lastest checkpoint to resume, the wandb log will automatically resume.
* **Efficient Training on Pre-Extracted Latent:** Set ```use_latent``` to ```True```, and set the ```data_list``` to correponding latent path list.

## 💻 Sampling & Evaluation

To evaluate the performance of a pretrained model, just copy the training config and set the ```pretrain_network: ~``` to your trained folder. Then run the following scripts:


```bash
accelerate launch \
    --num_processes 8 \
    --num_machines 1 \
    --main_process_port 10410 \
    test.py \
    -opt test_config.yml
```

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📖 Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@article{gu2025long,
    title={Long-Context Autoregressive Video Modeling with Next-Frame Prediction},
    author={Gu, Yuchao and Mao, weijia and Shou, Mike Zheng},
    journal={arXiv preprint arXiv:2503.19325},
    year={2025}
}
```
