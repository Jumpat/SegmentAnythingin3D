# SA3D-GS

This is the official implementation of the 3D-GS version of SA3D (Segment Anything in 3D with Radiance Fields). With SA3D-GS, fine-grained 3D segmentation can be achieved within seconds.

<br>

# Installation
The installation of SA3D is similar to [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
```bash
git clone git@github.com:Jumpat/SegmentAnythingin3D.git
```
or
```bash
git clone https://github.com/Jumpat/SegmentAnythingin3D.git
```

```bash
cd SegmentAnythingin3D;
git checkout sa3d-gs
```

Then install the dependencies:
```bash
conda env create --file environment.yml
conda activate gaussian_splatting_sa3d
```

Install SAM:
```bash
cd third_party;
git clone git@github.com:facebookresearch/segment-anything.git 
cd segment-anything; pip install -e .
mkdir sam_ckpt; cd sam_ckpt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Prepare Data

The used datasets are [360_v2](https://jonbarron.info/mipnerf360/), [nerf_llff_data](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7) and [LERF](https://drive.google.com/drive/folders/1vh0mSl7v29yaGsxleadcj-LCZOE_WEWB?usp=sharing).

The data structure of SA3D-GS is shown as follows:
```
./data
    /360_v2
        /garden
            /images
            /sparse
            /features
        ...
    /nerf_llff_data
        /fern
            /images
            /poses_bounds.npy
            /sparse
            /features
        /horns
            ...
        ...
    /lerf_data
        ...
```

## Pre-train the 3D Gaussians
We inherit all attributes from 3DGS, more information about training the Gaussians can be found in their repo.
```bash
python train_scene.py -s <path to COLMAP or NeRF Synthetic dataset>
```

## 3D Segmentation
Before the segmentation phase, you need to extract the SAM encoder features, run the following command:
```bash
python extract_features.py --image_root <path to the scene data> --sam_checkpoint_path <path to the pre-trained SAM model> --downsample <1/2/4/8>
```

Then run the segmentation:
```bash
python train_seg.py -m <path to the pre-trained 3D-GS model>
```

Please note that currently we haven't implement a GUI yet, you have to change the hardcoded prompt points in *train_seg.py*.

## Rendering
After the segmentation, you can render the segmentation results by running the following command:
```bash
python render.py -m <path to the pre-trained 3DGS model> --target scene --segment
```

You can also render the pre-trained 3DGS model without segmentation:
```bash
python render.py -m <path to the pre-trained 3DGS model> --target scene
```

# Acknowledgement
The code is built based on [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Segment Anything](https://github.com/facebookresearch/segment-anything). We also borrow the implementation of 3D-GS depth rendering (i.e., submodule/diff-gaussian-rasterization-depth) from [here](https://github.com/ashawkey/diff-gaussian-rasterization). We thank them all for their great work.


# Citation
If you find this project helpful for your research, please consider citing our paper and giving a ⭐.

```BibTex
@inproceedings{cen2023segment,
      title={Segment Anything in 3D with NeRFs}, 
      author={Jiazhong Cen and Zanwei Zhou and Jiemin Fang and Chen Yang and Wei Shen and Lingxi Xie and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian},
      booktitle    = {NeurIPS},
      year         = {2023},
}
```
