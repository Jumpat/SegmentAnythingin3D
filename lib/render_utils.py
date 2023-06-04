import torch
from tqdm import tqdm, trange
import numpy as np
from .dvgo import get_rays_of_a_view
import os
import imageio
from .utils import to8b, rgb_lpips, rgb_ssim, gen_rand_colors
import matplotlib.pyplot as plt


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False, cfg=None,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False, 
                      seg_mask=True, render_fct=0.0, seg_type='seg_density'):
    '''Render images for the given viewpoints; run evaluation if gt given.'''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs, segs, depths, bgmaps, psnrs, ssims, lpips_alex, lpips_vgg = [], [], [], [], [], [], [], []

    for i, c2w in enumerate(tqdm(render_poses, desc='Render {}...'.format(seg_type))):
        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        if seg_mask: keys.append('seg_mask_marched')
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, render_fct=render_fct, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        
        rgb = render_result['rgb_marched'].cpu().numpy()
            
        if seg_mask:
            seg_m = render_result['seg_mask_marched'].cpu()
        else:
            seg_m = None
            
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        if seg_mask:
            segs.append(seg_m)
        depths.append(depth)
        bgmaps.append(bgmap)
        if i==0:
            print('Testing, rgb shape: ', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)
            segs[i] = np.flip(segs[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))
            segs[i] = np.rot90(segs[i], k=render_video_rot90, axes=(0,1))
            
    if savedir is not None and dump_images:
        if seg_type == 'seg_density':
            img_dir = 'seged_img'
        elif seg_type == 'seg_img':
            img_dir = 'ori_img'
        else:
            raise NotImplementedError
        img_dir = os.path.join(savedir, img_dir)
        os.makedirs(img_dir, exist_ok=True)
        for i in trange(len(rgbs), desc='dumping images...'):
            rgb8 = to8b(rgbs[i])
            filename = os.path.join(img_dir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)
    if len(segs): segs = np.stack(segs)

    return rgbs, depths, bgmaps, segs


def fetch_render_params(render_type, data_dict):
    if render_type == 'train':
        render_poses=data_dict['poses'][data_dict['i_train']]
        HW=data_dict['HW'][data_dict['i_train']]
        Ks=data_dict['Ks'][data_dict['i_train']]
        gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']]
    elif render_type == 'test':
        render_poses=data_dict['poses'][data_dict['i_test']]
        HW=data_dict['HW'][data_dict['i_test']]
        Ks=data_dict['Ks'][data_dict['i_test']]
        gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']]
    elif render_type == 'video':
        render_poses=data_dict['render_poses']
        HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0)
        Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0)
        gt_imgs=None
    else:
        raise NotImplementedError
    
    return render_poses, HW, Ks, gt_imgs
        

@torch.no_grad()
def render_fn(args, cfg, ckpt_name, flag, e_flag, num_obj, data_dict, render_viewpoints_kwargs, seg_type='seg_density'):
    rand_colors = gen_rand_colors(num_obj)
    testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_{args.render_opt}_{ckpt_name}')
    os.makedirs(testsavedir, exist_ok=True)
    print('All results are dumped into', testsavedir)
    render_poses, HW, Ks, gt_imgs = fetch_render_params(args.render_opt, data_dict)
    rgbs, depths, bgmaps, segs = render_viewpoints(
            render_poses=render_poses,
            HW=HW, Ks=Ks, gt_imgs=gt_imgs,
            cfg=cfg,savedir=testsavedir, dump_images=args.dump_images,
            eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
            seg_type=seg_type,
            **render_viewpoints_kwargs)
    
    imageio.mimwrite(os.path.join(testsavedir, 'video.rgb'+flag+e_flag+'_'+seg_type+'.mp4'), to8b(rgbs), fps=30, quality=8)
    imageio.mimwrite(os.path.join(testsavedir, 'video.seg'+flag+e_flag+'_'+seg_type+'.mp4'), to8b(segs>0), fps=30, quality=8)
    # imageio.mimwrite(os.path.join(testsavedir, 'video.depth'+flag+e_flag+'_'+seg_type+'.mp4'), \
    #                  to8b(1 - depths / np.max(depths)), fps=30, quality=8)
    depth_vis = plt.get_cmap('rainbow')(1 - depths / np.max(depths)).squeeze()[..., :3]
    imageio.mimwrite(os.path.join(testsavedir, 'video.depth'+flag+e_flag+'_'+seg_type+'.mp4'), to8b(depth_vis), fps=30, quality=8)
    if False:
        depths_vis = depths * (1-bgmaps) + bgmaps
        dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
        depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth'+flag+e_flag+'_'+seg_type+'.mp4'), to8b(depth_vis), fps=30, quality=8)

    if seg_type == 'seg_img':
        seg_on_rgb = []
        if args.dump_images:
            masked_img_dir = os.path.join(testsavedir, 'masked_img')
            os.makedirs(masked_img_dir, exist_ok=True)
        for i, rgb, seg in zip(range(rgbs.shape[0]), rgbs, segs):
            # Winner takes all
            max_logit = np.expand_dims(np.max(seg, axis = -1), -1)
            tmp_seg = seg
            tmp_seg = np.argmax(tmp_seg, axis = -1)
            tmp_seg[max_logit[:,:,0] <= 0.1] = num_obj
            recolored_rgb = 0.3*rgb + 0.7*(rand_colors[tmp_seg])
            seg_on_rgb.append(recolored_rgb)
            if args.dump_images:
                imageio.imwrite(os.path.join(masked_img_dir, 'rgb_{:03d}.png'.format(i)), to8b(recolored_rgb))
        imageio.mimwrite(os.path.join(testsavedir, 'video.seg_on_rgb'+e_flag+'_'+seg_type+'.mp4'), to8b(seg_on_rgb), fps=30, quality=8)
        return to8b(np.stack(seg_on_rgb))
    
    return to8b(rgbs)
