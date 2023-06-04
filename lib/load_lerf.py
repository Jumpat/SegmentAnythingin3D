import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

def normalize(x):
    return x / np.linalg.norm(x)

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).astype(np.float32)

trans_center = lambda centroid : np.array([
    [1,0,0,centroid[0]],
    [0,1,0,centroid[1]],
    [0,0,1,centroid[2]],
    [0,0,0,1]]).astype(np.float32)

rot_phi = lambda phi : np.array([ # rot dir: +y -> +z
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).astype(np.float32)

rot_theta = lambda th : np.array([ # rot dir: +x -> +z
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).astype(np.float32)

rot_gamma = lambda ga : np.array([ # rot dir: +x -> +y
    [np.cos(ga),-np.sin(ga),0,0],
    [np.sin(ga), np.cos(ga),0,0],
    [0,0,1,0],
    [0,0,0,1]]).astype(np.float32)


def pose_spherical(gamma, phi, t):
    c2w = np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]]).astype(np.float32)
    
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_gamma(gamma/180.*np.pi) @ c2w
    c2w[:3, 3] = t
    return c2w


def load_lerf_data(basedir, factor=2, args=None, movie_render_kwargs={}):
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        metas = json.load(fp)


    imgs = []
    poses = []
    intrinsics = []
    fts = []
    skip = 1

    for frame in metas['frames'][::skip]:
        fname = os.path.join(basedir, frame['file_path'])
        just_fname = fname.split('/')[-1]
        if factor >= 2:
            fname = os.path.join(basedir, 'images_{}'.format(factor), just_fname)
        else:
            fname = os.path.join(basedir, 'images', just_fname)
        imgs.append(imageio.imread(fname))
        poses.append(np.array(frame['transform_matrix']))
        K = np.array([
                [frame['fl_x']/factor, 0, frame['cx']/factor],
                [0, frame['fl_y']/factor, frame['cy']/factor],
                [0, 0, 1]
            ]).astype(np.float32)
        intrinsics.append(K)
    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)
    intrinsics = np.array(intrinsics).astype(np.float32)
    f_avg = (intrinsics[:, 0, 0] + intrinsics[:, 1, 1]).mean() / 2.

    i_test = np.arange(0, int(poses.shape[0]), 8)
    i_val = i_test
    i_train = np.array([i for i in np.arange(int(poses.shape[0])) if
                        (i not in i_test and i not in i_val)])
    i_split = [i_train, i_val, i_test]

    H, W = imgs[0].shape[:2]

    poses_ = poses.copy()
    centroid = poses_[:,:3,3].mean(0)
    radcircle = movie_render_kwargs.get('scale_r', 0) * np.linalg.norm(poses_[:,:3,3] - centroid, axis=-1).mean()
    centroid[0] += movie_render_kwargs.get('shift_x', 0)
    centroid[1] += movie_render_kwargs.get('shift_y', 0)
    centroid[2] += movie_render_kwargs.get('shift_z', 0)
    up_rad = movie_render_kwargs.get('pitch_deg', 0)
    # render_poses = torch.stack([pose_spherical(angle, up_rad, centroid) for angle in np.linspace(-180,180,80+1)[:-1]], 0)

    render_poses = []
    camera_o = np.zeros_like(centroid)
    num_render = 90
    for th in np.linspace(0., 360., num_render):
        camera_o[0] = centroid[0] + radcircle * np.cos(th/180.*np.pi)
        camera_o[1] = centroid[1] + radcircle * np.sin(th/180.*np.pi)
        camera_o[2] = centroid[2]
        render_poses.append(pose_spherical(th+90.0, up_rad, camera_o))
    render_poses = np.stack(render_poses, axis=0)

    return imgs, poses, render_poses, [H, W, f_avg], intrinsics, i_split
