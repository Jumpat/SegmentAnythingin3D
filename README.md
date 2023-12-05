# Segment Anything🤖️ in 3D with NeRFs (SA3D)
### [Project Page](https://jumpat.github.io/SA3D/) | [Arxiv Paper](https://arxiv.org/abs/2304.12308)

[Segment Anything in 3D with NeRFs](https://arxiv.org/abs/2304.12308)  
[Jiazhong Cen](https://github.com/Jumpat)\*<sup>1</sup>, [Zanwei Zhou](https://github.com/Zanue)\*<sup>1</sup>, [Jiemin Fang](https://jaminfong.cn/)<sup>2</sup>, [Chen Yang](https://github.com/chensjtu)<sup>1</sup>, [Wei Shen](https://shenwei1231.github.io/)<sup>1✉</sup>, [Lingxi Xie](http://lingxixie.com/)<sup>3</sup>, [Dongsheng Jiang](https://sites.google.com/site/dongshengjiangbme/)<sup>3</sup>, [Xiaopeng Zhang](https://sites.google.com/site/zxphistory/)<sup>3</sup>, [Qi Tian](https://scholar.google.com/citations?hl=en&user=61b6eYkAAAAJ)<sup>3</sup>   
<sup>1</sup>AI Institute, SJTU &emsp; <sup>2</sup>School of EIC, HUST &emsp; <sup>3</sup>Huawei Inc.  
\*denotes equal contribution  

*Given a NeRF, just input prompts from **one single view** and then get your 3D model.*   
<img src="imgs/SA3D.gif" width="800">

We propose a novel framework to Segment Anything in 3D, named <b>SA3D</b>. Given a neural radiance field (NeRF) model, SA3D allows users to obtain the 3D segmentation result of any target object via only <b>one-shot</b> manual prompting in a single rendered view. The entire process for obtaining the target 3D model can be completed in approximately 2 minutes, yet without any engineering optimization. Our experiments demonstrate the effectiveness of SA3D in different scenes, highlighting the potential of SAM in 3D scene perception. 

## Update
* **2023/11/11**: We release the nerfstudio version of SA3D. Currently it only supports the text prompt as input.
* **2023/06/29**: We now support [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) as the segmentation network. Follow the installation instruction in [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), and then download *mobile_sam.pt* into folder ``./dependencies/sam_ckpt``. You can use `--mobile_sam` to switch to MobileSAM.

## Overall Pipeline

![SA3D_pipeline](https://github.com/Jumpat/SegmentAnythingin3D/assets/58475180/6135f473-3239-4721-9a79-15f7a7d11347)

With input prompts, SAM cuts out the target object from the according view. The obtained 2D segmentation mask is projected onto 3D mask grids via density-guided inverse rendering. 2D masks from other views are then rendered, which are mostly uncompleted but used as cross-view self-prompts to be fed into SAM again. Complete masks can be obtained and projected onto mask grids. This procedure is executed via an iterative manner while accurate 3D masks can be finally learned. SA3D can adapt to various radiance fields effectively without any additional redesigning.


## Installation

```bash
git clone https://github.com/Jumpat/SegmentAnythingin3D.git
cd SegmentAnythingin3D

conda create -n sa3d python=3.10
pip install -r requirements.txt
```

### NeRFStudio
Follow [this guidance](https://docs.nerf.studio/quickstart/installation.html) to install nerfstudio.

Note: We developed our code under `nerfstudio==0.2.0`.

### SAM and Grounding-DINO:

```bash
cd sa3d/self_prompting; # now the folder 'dependencies' is under 'sa3d/self_prompting';

# Installing SAM
mkdir dependencies; cd dependencies 
mkdir sam_ckpt; cd sam_ckpt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
git clone git@github.com:facebookresearch/segment-anything.git 
cd segment-anything; pip install -e .

# Installing Grounding-DINO
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/; pip install -e .
mkdir weights; cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### SA3D
In the root directory of this repo, conduct
```bash
pip install -e .
```

## Usage
- Train NeRF
  ```bash
  ns-train nerfacto --load-data {data-dir}
  ```
- Run SA3D
  ```bash
  ns-train sa3d --data {data-dir} \
    --load-dir {ckpt-dir} \
    --pipeline.text_prompt {text-prompt} \
    --pipeline.network.num_prompts {num-prompts} \
  ```
- Render and Save Fly-through Videos
  ```bash
  ns-viewer --load-config {config-dir}
  ```

## Some Visualization Samples

SA3D can handle various scenes for 3D segmentation. Find more demos in our [project page](https://jumpat.github.io/SA3D/).


## Acknowledgements
Thanks for the following project for their valuable contributions:
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [DVGO](https://github.com/sunset1995/DirectVoxGO)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO.git)
- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)

## Citation
If you find this project helpful for your research, please consider citing the report and giving a ⭐.
```BibTex
@inproceedings{cen2023segment,
      title={Segment Anything in 3D with NeRFs}, 
      author={Jiazhong Cen and Zanwei Zhou and Jiemin Fang and Chen Yang and Wei Shen and Lingxi Xie and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian},
      booktitle    = {NeurIPS},
      year         = {2023},
}
```
