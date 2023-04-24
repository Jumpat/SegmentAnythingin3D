# Segment Anything🤖️ in 3D with NeRFs (SA3D)
## Work in progress
### [Project Page](https://jumpat.github.io/SA3D/) | [Arxiv Paper]()

This repo is for the official code of [SA3D](). 
<img src="imgs/SA3D.gif" width="800">

The [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) has demonstrated its effectiveness in segmenting any object/part in various 2D images, yet its ability for 3D has not been fully explored. The real world is composed of numerous 3D scenes and objects. Due to the scarcity of accessible 3D data and high cost of its acquisition and annotation, lifting SAM to 3D is a challenging but valuable research avenue. With this in mind, we propose a novel framework to Segment Anything in 3D, named <b>SA3D</b>. Given a neural radiance field (NeRF) model, SA3D allows users to obtain the 3D segmentation result of any target object via only <b>one-shot</b> manual prompting in a single rendered view. With input prompts, SAM cuts out the target object from the according view. The obtained 2D segmentation mask is projected onto 3D mask grids via density-guided inverse rendering. 2D masks from other views are then rendered, which are mostly uncompleted but used as cross-view self-prompts to be fed into SAM again. Complete masks can be obtained and projected onto mask grids. This procedure is executed via an iterative manner while accurate 3D masks can be finally learned. SA3D can adapt to various radiance fields effectively without any additional redesigning. The entire segmentation process can be completed in approximately two minutes without any engineering optimization. Our experiments demonstrate the effectiveness of SA3D in different scenes, highlighting the potential of SAM in 3D scene perception. The code will be released.

## Overall pipeline

<img src="imgs/SA3D_pipeline.png" width="800">

## Some visualization samples

SA3D can handle various scenes of 3D segmentation. More demos can be found in our [project page](https://jumpat.github.io/SA3D/).

| Forward facing | 360 degree| Multi-objects|
| :---: | :---:| :---:|
|<img src="imgs/horns.gif" width="200"> | <img src="imgs/lego.gif" width="200"> | <img src="imgs/orchid_multi.gif" width="200">

## Acknowledgements
We would like to acknowledge the following projects for their valuable contributions:
- [Segment Anything](https://github.com/facebookresearch/segment-anything)

## Citation
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```BibTex
```

