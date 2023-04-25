# Segment Anything🤖️ in 3D with NeRFs (SA3D)
## Work in Progress
### [Project Page](https://jumpat.github.io/SA3D/) | [Arxiv Paper](https://arxiv.org/abs/2304.12308)

[Segment Anything in 3D with NeRFs](https://arxiv.org/abs/2304.12308)  
[Jiazhong Cen](https://github.com/Jumpat)<sup>1</sup>, [Zanwei Zhou](https://github.com/Zanue)<sup>1</sup>, [Jiemin Fang](https://jaminfong.cn/)<sup>2</sup>, [Wei Shen](https://shenwei1231.github.io/)<sup>1✉</sup>, [Lingxi Xie](http://lingxixie.com/)<sup>3</sup>, [Dongsheng Jiang]()<sup>3</sup>, [Xiaopeng Zhang](https://sites.google.com/site/zxphistory/)<sup>3</sup>, [Qi Tian](https://scholar.google.com/citations?hl=en&user=61b6eYkAAAAJ)<sup>3</sup>   
<sup>1</sup>AI Institute, SJTU &emsp; <sup>2</sup>School of EIC, HUST &emsp; <sup>3</sup>Huawei Inc.

*Given a NeRF, just input prompts from **one single view** and then get your 3D model.*   
<img src="imgs/SA3D.gif" width="800">

We propose a novel framework to Segment Anything in 3D, named <b>SA3D</b>. Given a neural radiance field (NeRF) model, SA3D allows users to obtain the 3D segmentation result of any target object via only <b>one-shot</b> manual prompting in a single rendered view. The entire process for obtaining the target 3D model can be completed in approximately 2 minutes, yet without any engineering optimization. Our experiments demonstrate the effectiveness of SA3D in different scenes, highlighting the potential of SAM in 3D scene perception. 

*The code will be released.*

## Overall Pipeline

<img src="imgs/SA3D_pipeline.png" width="800">

With input prompts, SAM cuts out the target object from the according view. The obtained 2D segmentation mask is projected onto 3D mask grids via density-guided inverse rendering. 2D masks from other views are then rendered, which are mostly uncompleted but used as cross-view self-prompts to be fed into SAM again. Complete masks can be obtained and projected onto mask grids. This procedure is executed via an iterative manner while accurate 3D masks can be finally learned. SA3D can adapt to various radiance fields effectively without any additional redesigning.

## Some Visualization Samples

SA3D can handle various scenes for 3D segmentation. Find more demos in our [project page](https://jumpat.github.io/SA3D/).

| Forward facing | 360° | Multi-objects |
| :---: | :---:| :---:|
|<img src="imgs/horns.gif" width="200"> | <img src="imgs/lego.gif" width="200"> | <img src="imgs/orchid_multi.gif" width="200">

## Acknowledgements
Thanks for the following project for their valuable contributions:
- [Segment Anything](https://github.com/facebookresearch/segment-anything)

## Citation
If you find this project helpful for your research, please consider citing the report and giving a ⭐.
```BibTex
@article{cen2023segment,
      title={Segment Anything in 3D with NeRFs}, 
      author={Jiazhong Cen and Zanwei Zhou and Jiemin Fang and Wei Shen and Lingxi Xie and Xiaopeng Zhang and Qi Tian},
      journal={arXiv:2304.12308},
      year={2023}
}
```

