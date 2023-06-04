import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=5, args=None):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    if args is not None and args.distill_active:
        all_fts = []
    counts = [0]

    # get H, W
    tmp_img = imageio.imread(os.path.join(basedir, next(iter(metas.values()))['frames'][::1][0]['file_path'] + '.png'))
    H, W = tmp_img.shape[:2]
    if args is not None and args.distill_active:
        fts_dict = load_features(file=os.path.join(basedir, "features.pt"), imhw=(H, W))

    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        fts = []
        if s=='train' or testskip==0:
            skip = 3
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            just_fname = fname.split('/')[-1]
            if args is not None and args.distill_active:
                fts.append(fts_dict[just_fname].permute(1, 2, 0))
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        if args is not None and args.distill_active:
            fts = torch.stack(fts)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        if args is not None and args.distill_active:
            all_fts.append(fts)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    if args is not None and args.distill_active:
        fts = torch.cat(all_fts, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    if args is not None and args.distill_active:
        return imgs, poses, render_poses, [H, W, focal], i_split, fts
    else:
        return imgs, poses, render_poses, [H, W, focal], i_split, None
