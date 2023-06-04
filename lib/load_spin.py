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


def load_spin_data(basedir, spin_basedir, factor=None):

    spin_annotation_paths = os.listdir(spin_basedir)
    spin_annotation_paths = [n for n in spin_annotation_paths if 'cutout' not in n and 'pseudo' not in n and 'png' in n]
    
    
#     spin_annotation_paths = [os.path.join(spin_basedir, n) for n in spin_annotation_paths]
#     spin_annotation_paths = sorted(spin_annotation_paths)
   
    if factor is None:
        sfx = '_4'
    else:
        sfx = '_'+str(factor)

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir) and 'Truck' in imgdir:
        imgdir = os.path.join(basedir, 'train', 'rgb')
    elif not os.path.exists(imgdir) and 'lego' in imgdir:
        imgdir = os.path.join(basedir, 'rgb')
        from skimage.transform import resize
    elif not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    sorted_image_names = [f for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    
    id_to_gt_mask = {}
    ref_id = None
    for spin_annotation_name in spin_annotation_paths:
        for i in range(len(sorted_image_names)):
            if sorted_image_names[i].split('.')[-2] in spin_annotation_name.split('.')[-2]:
                if 'lego' in imgdir:
                    tmp = resize(imageio.imread(os.path.join(spin_basedir, spin_annotation_name)).astype(np.float32), (768, 1020))
                    tmp[tmp >= 0.5] = 1
                    tmp[tmp != 1] = 0
                    id_to_gt_mask[i] = tmp
#                     id_to_gt_mask[i] = imageio.imread(os.path.join(spin_basedir, spin_annotation_name))
#                     print(np.unique(id_to_gt_mask[i]), "??")
                else:
                    id_to_gt_mask[i] = imageio.imread(os.path.join(spin_basedir, spin_annotation_name))
                if ref_id is None:
                    ref_id = i
                break
    
    return ref_id, id_to_gt_mask

# if __name__ == '__main__':
#     print(load_spin_data('/datasets/nerf_data/nerf_llff_data/room/', '/datasets/nerf_data/MVSeg_data/room/'))