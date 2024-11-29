# Diptych

<!-- ![](assets/teaser.png) -->

Code implementation of "Large-Scale Text-to-Image Model with Inpainting is a Zero-Shot Subject-Driven Image Generator" [[Link](https://arxiv.org/pdf/2411.15466)] 

<!-- Will appear at CVPR 2024! -->

<!-- ## Abstact

Concept personalization methods enable large text-to-image models to learn specific subjects (e.g., objects/poses/3D models) and synthesize renditions in new contexts. Given that the image references are highly biased towards visual attributes, state-of-the-art personalization models tend to overfit the whole subject and cannot disentangle visual characteristics in pixel space. In this study, we proposed a more challenging setting, namely fine-grained visual appearance personalization. Different from existing methods, we allow users to provide a sentence describing the desired attributes. A novel decoupled self-augmentation strategy is proposed to generate target-related and non-target samples to learn user-specified visual attributes. These augmented data allow for refining the model's understanding of the target attribute while mitigating the impact of unrelated attributes. At the inference stage, adjustments are conducted on semantic space through the learned target and non-target embeddings to further enhance the disentanglement of target attributes. Extensive experiments on various kinds of visual attributes with SOTA personalization methods show the ability of the proposed method to mimic target visual appearance in novel contexts, thus improving the controllability and flexibility of personalization. -->

<!-- ## Pipeline

![](assets/pipeline.png) -->

## Setup

Clone this project and install dependencies to set up the environment (Python 3.11 is recommended):
```
cd Diptych
pip install -r requirements.txt
```
Prepare GroundingDINO: 
```
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```
Prepare SAM: 
```
mkdir SAM_checkpoints
```
Then download required checkpoints from: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) under ```./SAM_checkpoints/```.

## Running

```
mkdir output
python inference_diptych.py --arg1 * --arg2 *
```

## Citation

```BibTeX
@article{shin2024large,
  title={Large-Scale Text-to-Image Model with Inpainting is a Zero-Shot Subject-Driven Image Generator},
  author={Shin, Chaehun and Choi, Jooyoung and Kim, Heeseung and Yoon, Sungroh},
  journal={arXiv preprint arXiv:2411.15466},
  year={2024}
}
```

## Acknowledgements

The code is mainly based on [diffusers](https://github.com/huggingface/diffusers) and [FLUX-Controlnet-Inpainting](https://github.com/alimama-creative/FLUX-Controlnet-Inpainting).