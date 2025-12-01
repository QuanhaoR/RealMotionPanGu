# RealMotion improved based on the openPangu model


> [**Motion Forecasting in Continuous Driving**](https://arxiv.org/abs/2410.06007)            
> Nan Song, [Bozhou Zhang](https://zbozhou.github.io/), [Xiatian Zhu](https://surrey-uplab.github.io/), [Li Zhang](https://lzrobots.github.io)   
> **NeurIPS 2024**

> [**Frozen Transformers in Language Models Are Effective Visual Encoder Layers**](https://arxiv.org/abs/2310.12973)            
>[Ziqi Pang](https://ziqipang.github.io/), [Ziyang Xie*](https://ziyangxie.site/), [Yunze Man*](https://yunzeman.github.io/), [Yu-Xiong Wang](https://yxw.web.illinois.edu/)   
> **ICLR 2024**

## 🚗 Abstract
Motion forecasting for agents in autonomous driving is highly challenging due to the stochastic nature of future actions and complex spatiotemporal interactions. While the RealMotion framework successfully addresses the limitations of independent scene processing through its dual-stream architecture (scene context and agent trajectory streams), optimizing the representation of these continuous high-level features remains a critical avenue for improvement. Inspired by recent findings that Large Language Models (LLMs) possess surprising efficacy as encoders for non-semantic tasks, we propose an enhanced motion forecasting framework that integrates the **openPangu** model into the RealMotion backbone. Specifically, we insert a frozen transformer block from the pre-trained **Pangu** LLM immediately following the RealMotion encoder. By treating the encoder's output as visual tokens, the frozen **Pangu** block acts as a sophisticated information filter, discerning informative spatiotemporal features and amplifying their activation while suppressing noise—all without the need for textual prompts or multimodal alignment. This approach effectively leverages the generalized pattern-recognition capabilities of LLMs to refine the continuous context captured by RealMotion. Extensive experiments on the Argoverse benchmarks demonstrate that our method outperforms the original RealMotion baseline, validating the effectiveness of employing frozen LLM layers to enhance continuous motion forecasting.


## ⭐ Model source
This project incorporates the last layer of the [**openPangu-Embedded-1B-V1.1**](https://gitcode.com/ascend-tribe/openPangu-Embedded-1B-V1.1) model as a core component (feature encoder). The model was originally developed and open-sourced by **Huawei** and the **OpenPangu Team**.

* **Original Model**: openPangu-Embedded-1B-V1.1
* **Developer**: Huawei Technologies Co., Ltd. & OpenPangu Team
* **Architecture Note**: We utilize the Transformer blocks of OpenPangu as a frozen visual encoder. The pre-trained weights are used in accordance with the open-source license.

## 🛠️ Get started
### 1. Development Environment
This project was developed and validated on a server with the following specifications:

- CPU Architecture: aarch64 (ARM, Huawei Kunpeng processor)

- AI Accelerator: Huawei Ascend 910B2 NPU

- CANN Version: 8.1.RC1

### 2.Set up a new virtual environment
```
conda create -n motion python=3.10.9
conda activate motion
```

### 3.Install dependency packpages
```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### 4.Replace PyTorch Lightning Packages for Ascend NPU Compatibility:
After completing the above steps, please navigate to miniconda3/envs/motion/lib/python3.10/site-packages/ and replace the lightning_fabric, lightning_utilities, and pytorch_lightning packages with the three corresponding packages from https://github.com/Aurora-1412/ascend_npu_for_pytorch_lightning.

## 🕹️ Prepare the data
### 1. Setup [Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html)
```
data
    ├── train
    │   ├── 0000b0f9-99f9-4a1f-a231-5be9e4c523f7
    │   ├── 0000b6ab-e100-4f6b-aee8-b520b57c0530
    │   ├── ...
    ├── val
    │   ├── 00010486-9a07-48ae-b493-cf4545855937
    │   ├── 00062a32-8d6d-4449-9948-6fedac67bfcd
    │   ├── ...
    ├── test
    │   ├── 0000b329-f890-4c2b-93f2-7e2413d4ca5b
    │   ├── 0008c251-e9b0-4708-b762-b15cb6effc27
    │   ├── ...
```

### 2. Preprocess
```
python preprocess.py -d /path/to/data -p
```

### 3. The structure of the dataset after processing
```
└── data
    └── realmotion_processed
        ├── train
        ├── val
        └── test
```
## Loading PanGu Model Weights
### Steps:

- Download the weights from: [openPangu-Embedded-1B-V1.1](https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-1B-V1.1/tree/main)

- Create a directory named pangu_ckpt in your project root.

- Place the downloaded weight file (model.safetensors) inside the pangu_ckpt directory.

Your final directory structure should look like this:
```
RealMotionPanGu/
├── pangu_ckpt/
│   └── model.safetensors
├── train_pangu.py
├── requirements.txt
└── ...
```

## 🔥 Training and testing
```
# Train
python train.py

# Val
python eval.py checkpoint=/path/to/ckpt

# Test for submission
python eval.py checkpoint=/path/to/ckpt submit=true
```
## 🎞️ License
The usage of the model weights and related code in this project is subject to the **[OpenPangu Model License Agreement Version 1.0](https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-1B-V1.1/blob/main/LICENSE)**.By downloading, copying, using, modifying, or distributing any part of this project, you agree to be bound by the terms of this license agreement.

## ❤️ Acknowledgements
 - [openPangu](https://gitcode.com/ascend-tribe)
 - [Forecast-MAE](https://github.com/jchengai/forecast-mae)
 - [StreamPETR](https://github.com/exiawsh/StreamPETR)
 - [DeMo (Ours)](https://github.com/fudan-zvg/DeMo)
 - [LM4VisualEncoding](https://github.com/ziqipang/LM4VisualEncoding)