import numpy as np
import os, imageio
import torch
import scipy
from tqdm import tqdm

########## Slightly modified version of LLFF data loading code
##########  see https://github.com/Fyusion/LLFF for original
def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def load_nvos_data(basedir, factor=8):

    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    if poses_arr.shape[1] == 17:
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    elif poses_arr.shape[1] == 14:
        poses = poses_arr[:, :-2].reshape([-1, 3, 4]).transpose([1,2,0])
    else:
        raise NotImplementedError
    bds = poses_arr[:, -2:].transpose([1,0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

#     if factor is not None and factor != 1:
#         sfx = '_{}'.format(factor)
#         _minify(basedir, factors=[factor])
#         factor = factor
#     else:
#         factor = 1
    

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    
    # the name of scene, e.g. horns, orchids, trex, ...    
    scene_name = basedir.split('/')[-1]
    prefix_path = basedir[:-len(scene_name)]
    
    if 'horns' in scene_name:
        scene_name = 'horns_left' if int(input("Please choose the segmentation target of horns: 0. left; 1. center")) == 0 else 'horns_center'
       
    # get ref name
    ref_name_pre = os.path.join(prefix_path, 'reference_image', scene_name)
    ref_name = os.listdir(ref_name_pre)[0]
    
    # get target name and mask path
    target_name_pre = os.path.join(prefix_path, 'masks', scene_name)
    target_names = os.listdir(target_name_pre)
    
    mask_path = None
    for name in target_names:
        if "_mask" in name:
            mask_path = os.path.join(prefix_path, 'masks', scene_name, name)
        else:
            target_name = name
            
#     print("mask_path", mask_path)
#     print("target_name", target_name)
#     print("ref_name", ref_name)
    
    # get reference image index
    ref_ind = -1
    for ind, img_name in enumerate(imgfiles):
#         print(ind, img_name)
        if ref_name in img_name:
            ref_ind = ind
            break
    assert ref_ind != -1 and "no available reference image"
    
    # get target image index
    target_ind = -1
    for ind, img_name in enumerate(imgfiles):
        
        if target_name in img_name:
            target_ind = ind
            break
    assert target_ind != -1 and "no available target image"
    
    # load target mask
    target_mask = imageio.imread(mask_path)/255.
    
    # load scribbles
    scribbles_path = os.path.join(prefix_path, 'scribbles', scene_name)
    pos_path, neg_path = None, None
    for scribble_name in os.listdir(scribbles_path):
        if 'pos' in scribble_name:
            pos_path = os.path.join(scribbles_path, scribble_name)
        elif 'neg' in scribble_name:
            neg_path = os.path.join(scribbles_path, scribble_name)
            
    
    pos_scribbles = imageio.imread(pos_path)/255.
    neg_scribbles = imageio.imread(neg_path)/255.
    
    if len(pos_scribbles.shape) == 3:
        pos_scribbles = pos_scribbles.sum(-1)
        pos_scribbles[pos_scribbles != 0] = 1
    if len(neg_scribbles.shape) == 3:
        neg_scribbles = neg_scribbles.sum(-1)
        neg_scribbles[neg_scribbles != 0] = 1
    
    print("Skeletonizing the NVOS Scribbles")
    from skimage import morphology
    pos_scribbles = morphology.skeletonize(pos_scribbles).astype(np.float32)
    neg_scribbles = morphology.skeletonize(neg_scribbles).astype(np.float32)
        
    pos_scribbles *= np.random.rand(pos_scribbles.shape[0], pos_scribbles.shape[1])
    neg_scribbles *= np.random.rand(pos_scribbles.shape[0], pos_scribbles.shape[1])
    
    pos_scribbles[pos_scribbles < 0.98] = 0
    neg_scribbles[neg_scribbles < 0.995] = 0
    
    sh = imageio.imread(imgfiles[0]).shape
    if poses.shape[1] == 4:
        poses = np.concatenate([poses, np.zeros_like(poses[:,[0]])], 1)
        poses[2, 4, :] = np.load(os.path.join(basedir, 'hwf_cxcy.npy'))[2]
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor

    
    target_pose = poses[:,:,target_ind]
    ref_pose = poses[:,:,ref_ind]
    
    pos_points = np.where(pos_scribbles)
    pos_points = np.concatenate([pos_points[1][:, None], pos_points[0][:, None]], axis = 1)
    
    neg_points = np.where(neg_scribbles)
    neg_points = np.concatenate([neg_points[1][:, None], neg_points[0][:, None]], axis = 1)

    return ref_ind, ref_pose, pos_points, neg_points, target_ind, target_pose, target_mask