import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from tqdm import tqdm
import imageio
import torch
import torch.nn.functional as F



def mkdir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)

def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file.
    """
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n"%(
        xyz[i, 0], xyz[i, 1], xyz[i, 2],
        rgb[i, 0], rgb[i, 1], rgb[i, 2],
        ))

class Reprojector:
    '''
    Transform pixels in target views to points in world, and project them to source views.

    We use the opencv/COLMAP format of pose, +X right, +Y down, +Z front. 
    When the input pose is opengl/llff format, a transform matrix is need.
    '''
    def __init__(self, H, W, map_is_distance=True, \
                 input_format='opengl', coor_mode='center', batch_size=20, \
                    delta=1e-2, mask_rgb=[250.,0,0], device="cuda:0"):
        self.H, self.W = H, W
        self.map_is_distance = map_is_distance
        self.coor_mode = coor_mode
        self.device = device
        self.batch_size = batch_size # batch source views
        self.delta = delta
        self.mask_rgb = np.array(mask_rgb).reshape(1, 3)

        X, Y = torch.meshgrid(torch.arange(self.W), torch.arange(self.H), indexing='xy')
        X, Y = X.to(device), Y.to(device)
        if self.coor_mode == 'center':
            X, Y = X+0.5, Y+0.5
        self.X, self.Y = X, Y

        self.transfrom_matrix = None
        if input_format == 'opengl':
            self.transfrom_matrix = torch.tensor([
                [1., 0., 0.],
                [0., -1., 0.],
                [0., 0., -1.]
            ], dtype=torch.float32, device=device)

    
    @staticmethod
    def rescale_pose(pose, scene_center, scene_radius):
        '''
        Rescale pose matrix by scene center and radius.
        '''
        scene_center = scene_center.view(1, 3)
        scene_radius = scene_radius.view(1, 3)
        pose[:, :3, 3] = (pose[:, :3, 3] - scene_center) / scene_radius

        return pose

    def inbound_pixel(self, xy):
        '''
        Check if per coord is in the valid pixel range.

        Args:
        xy: pixel coordinates, shape of [num_views, 1, h*w]

        Return:
        xy_mask: mask of xy, shape of [num_views, 1, h*w]
        '''
        bound = np.array([0, self.W-1, 0, self.H-1])
        if self.coor_mode == 'center':
            bound = bound + 0.5
        xy_mask = (xy[:, 0:1] <= bound[1]) & (xy[:, 0:1] >= bound[0]) & (xy[:, 1:2] <= bound[3]) & (xy[:, 1:2] >= bound[2])

        return xy_mask
    
    def pixel2world(self, pose, intrinsic, distance_map):
        '''
        Transform 2d pixels to be 3d points in world.

        Args:
        pose: input pose matrix, shape of [tgt_views, 3, 4]
        intrinsic: input intrinsic, [tgt_views, 3, 3]
        distance_map: [tgt_views, H, W, 1]
        map_is_distance: if False, distance_map represents the depth (z direction)

        Return:
        points_world: [tgt_views, 3, H*W]
        '''
        
        tgt_views = pose.shape[0]
        X, Y = self.X[None].repeat(tgt_views, 1, 1), self.Y[None].repeat(tgt_views, 1, 1) # [tgt_views, H, W]

        # if inverse_y: # dirs: [3, H, W]
        #     dirs = torch.stack([(X-intrinsic[0][2])/intrinsic[0][0], (Y-intrinsic[1][2])/intrinsic[1][1], torch.ones_like(X)], 0)
        # else:
        #     dirs = torch.stack([(X-intrinsic[0][2])/intrinsic[0][0], -(Y-intrinsic[1][2])/intrinsic[1][1], -torch.ones_like(X)], 0)
        dirs = torch.stack([(X-intrinsic[:, :1, 2:3])/intrinsic[:, :1, :1], (Y-intrinsic[:, 1:2, 2:3])/intrinsic[:, 1:2, 1:2], torch.ones_like(X)], 1)
        dirs = dirs.reshape(tgt_views, 3, self.H*self.W) # dirs: [tgt_views, 3, H*W]

        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.matmul(pose[:, :3, :3], dirs) # [tgt_views, 3, H*W]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = pose[:,:3,3:4].expand(rays_d.shape) # [tgt_views, 3, H*W]

        # distance map refers to the point2point distance, while depth map refers to the absolute value of z.
        if self.map_is_distance:
            rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)
        distance_map = distance_map.reshape(tgt_views, self.H*self.W, 1).permute(0, 2, 1) # [tgt_views, 1, H*W]
        points_world = rays_o + distance_map * rays_d

        return points_world
    
    def world2pixel(self, points_world_, pose, intrinsic):
        '''
        Project target 3d points in world to souce views to be 2d pixels.

        Args:
        points_world_: shape of [tgt_views, 3, h*w]
        pose: poses to be projected, shape of [src_views, 3, 4]
        intrinsic: source intrinsics, [src_views, 3, 3]

        Return:
        xy: pixel coords, shape of [tgt_views*src_views, 2, h*w]
        depth: reprojected depth, shape of [tgt_views*src_views, 1, h*w]
        '''
        points_world = points_world_.clone()
        tgt_views, src_views = points_world.shape[0], pose.shape[0]
        points_world = points_world.repeat_interleave(src_views, dim=0) # [tgt_views*src_views, 3, h*w]
        c2w = pose[:, :3, :3].repeat(tgt_views, 1, 1) # [tgt_views*src_views, 3, 3]

        rays_o = pose[:, :3, 3:4].repeat(tgt_views, 1, 1) # [tgt_views*src_views, 3, 1]
        points_cam = torch.matmul(c2w.mT, points_world - rays_o) # [tgt_views*src_views, 3, h*w]
        intrinsic = intrinsic.repeat(tgt_views, 1, 1) # [tgt_views*src_views, 3, 3]
        coords = torch.matmul(intrinsic, points_cam) # [tgt_views*src_views, 3, h*w]
        depth = coords[:, -1:]
        xy = coords[:, :2] / depth

        return xy, depth

    def get_depth_mask(self, xy, xy_mask, depth_src, depth_reprojected, delta=1e-1, interpolation='bilinear'):
        '''
        Check if a 3d point is occluded. Similar to shadow mapping. 

        Args:
        xy: pixel coords, shape of [tgt_views*src_views, h, w, 2]
        xy_mask: inbounded coords, shape of [tgt_views*src_views, h, w]
        depth_src: shape of [src_views, h, w, 1]
        depth_reprojected: shape of [tgt_views*src_views, h, w, 1]

        Return:
        depth_mask: shape of [tgt_views*src_views, h, w]
        '''
        tgt_src = xy.shape[0]
        if interpolation == 'bilinear':
            xy_norm = torch.zeros_like(xy)
            xy_norm[..., :1] = xy[..., :1] * 2 / (self.W - 1) - 1 
            xy_norm[..., 1:] = xy[..., 1:] * 2 / (self.H - 1) - 1
            if self.coor_mode == 'center':
                xy_norm[..., :1] -= 1 / (self.W - 1)
                xy_norm[..., 1:] -= 1 / (self.H - 1)
            src_views = depth_src.shape[0]
            depth_src = depth_src.repeat(tgt_src // src_views, 1,1,1).permute(0,3,1,2) # [tgt_views*src_views, 1, H, W]
            depth_sampled = F.grid_sample(depth_src, xy_norm, mode='bilinear', padding_mode='zeros', align_corners=True).permute(0,2,3,1) # [tgt_views*src_views, H, W, 1]
            depth_sampled = depth_sampled[xy_mask].reshape(-1)
        # elif interpolation == 'nearest':
        #     xy_projected = xy[xy_mask.repeat_interleave(2, dim=-1)].reshape(tgt_src, -1, 2)
        #     xy_projected = torch.round(xy_projected)
        #     depth_sampled = depth_src[list(xy_projected[..., 1]), list(xy_projected[..., 0])].reshape(tgt_src, -1, 1) # note: xy in image is the opposite to that in array
        depth_mask = torch.zeros((tgt_src*self.H*self.W,), dtype=torch.bool).to(xy.device)
        depth_mask[xy_mask.reshape(-1)] = (depth_reprojected[xy_mask].reshape(-1) <= ((1+delta)*depth_sampled))
        depth_mask = depth_mask.reshape(tgt_src, self.H, self.W)

        return depth_mask
    
    @torch.no_grad()
    def reproject(self, tgt_pose, src_pose, tgt_depth, src_depth, tgt_intrinsic, src_intrinsic, valid_mask=None, \
                  tgt_rgb=None, src_rgb=None, new_img_paths=None, save_dir=None):
        '''
        Main function, which transforms pixels in target views to points in world, and then project them to source views.

        Args:
        valid_mask: shape of [tgt_views, h, w]
        tgt_pose: shape of [tgt_views, 3, 4]
        src_pose: shape of [src_views, 3, 4]
        tgt_depth: shape of [tgt_views, h, w, 1]
        src_depth: shape of [src_views, h, w, 1]
        tgt_intrinsic: shape of [tgt_views, 3, 3]
        src_intrinsic: shape of [src_views, 3, 3]

        Return:
        src_view_count: shape of [tgt_views, src_views], representing how many pixels are projected into j^th src_view from i^th tgt_view.

        '''
        assert(tgt_pose.shape[0] == tgt_depth.shape[0])
        tgt_views, src_views = tgt_pose.shape[0], src_pose.shape[0]
        device = self.device
        h, w = self.H, self.W
        if valid_mask is None:
            valid_mask = torch.ones_like(tgt_depth[:-1])
        valid_mask = valid_mask.to(device)
        src_view_count = torch.zeros((tgt_views, src_views), device=device) # (h, w) = (800, 800)

        tgt_intrinsic = tgt_intrinsic.float().to(device)
        tgt_pose = tgt_pose.float().to(device)
        tgt_depth = tgt_depth.float().to(device)
        src_intrinsic = src_intrinsic.float().to(device)
        src_pose = src_pose.float().to(device)
        if src_depth is not None:
            src_depth = src_depth.float().to(device)

        # project 2d to 3d
        points_world = self.pixel2world(tgt_pose, tgt_intrinsic, tgt_depth) # [tgt_views, 3, H*W]

        # for debug func pixel2world
        if False and save_dir is not None:
            pts_rgb = tgt_rgb.copy()
            points_world_ = points_world.clone()
            src_pose_ = src_pose[:22].float().to(device)
            src_depth_ = src_depth[:22].float().to(device)
            src_intrinsic_ = src_intrinsic[:22].float().to(device)
            for i in range(20, 22):
                points_world_ = torch.cat([points_world_, self.pixel2world(src_pose_[i:i+1], src_intrinsic_[i:i+1], src_depth_[i:i+1])], dim=0)
                pts_rgb = np.concatenate([pts_rgb, src_rgb[i:i+1]], axis=0)
            points_world_numpy = points_world_.permute(0,2,1).cpu().numpy() # [src_views, H*W, 3]
            points_cloud = np.concatenate([points_world_numpy.reshape(-1,3), pts_rgb[..., :3].reshape(-1,3)], axis=-1)
            pcwrite(os.path.join(save_dir, 'pc.ply'), points_cloud)
            print('point clouds written!')
        

        save_rgbs = []
        idx = 0
        for batch_src_pose, batch_src_intrinsic in zip(src_pose.split(self.batch_size, 0), src_intrinsic.split(self.batch_size, 0)):
            batch_src_views = batch_src_pose.shape[0]
            if src_depth is not None:
                batch_src_depth = src_depth[idx:idx+batch_src_views]
            else:
                batch_src_depth = None
            

            # project 3d to 2d
            xy, depth_reprojected = self.world2pixel(points_world, batch_src_pose, batch_src_intrinsic) # [tgt_views*src_views, 2/1, h*w]
            
            # check pixel bound
            xy_mask = self.inbound_pixel(xy) # [tgt_views*src_views, 1, h*w]
            front_mask = (depth_reprojected > 0) # [tgt_views*src_views, 1, h*w]
            xy_mask = (xy_mask & front_mask)
            xy = xy.permute(0,2,1).reshape(-1, h, w, 2) # [tgt_views*src_views, h, w, 2]
            depth_reprojected = depth_reprojected.reshape(-1, h, w, 1) # [tgt_views*src_views, h, w, 1]
            xy_mask = xy_mask.reshape(-1, h, w) # [tgt_views*src_views, h, w]

            if batch_src_depth is None: # No occlusion check
                depth_mask = (torch.ones_like(xy_mask) > 0)
            else: # check occlusion
                depth_mask = self.get_depth_mask(xy, xy_mask, batch_src_depth, depth_reprojected, delta=self.delta) # [tgt_views*src_views, H, W]
            
            # batch_tgt_src_mask = xy_mask # for debug
            # batch_tgt_src_mask = (xy_mask & depth_mask) # [tgt_views*src_views, H, W]
            batch_tgt_src_mask = (xy_mask & depth_mask) & (valid_mask >= 0.5).repeat_interleave(batch_src_views, dim=0) # [tgt_views*src_views, H, W]
            # count
            src_view_count[:, idx:idx+batch_src_views] = batch_tgt_src_mask.reshape(tgt_views, batch_src_views, h*w).sum(-1) # [tgt_views, src_views]
            
            
            if save_dir is not None:
                batch_tgt_src_mask = batch_tgt_src_mask.cpu().numpy().reshape(tgt_views, batch_src_views, h, w)
                xy = xy.cpu().numpy().reshape(tgt_views, batch_src_views, h, w, 2)
                rgbs = src_rgb[idx:idx+batch_src_views]
                for i in range(tgt_views): # we only test when tgt_views==1
                    for j in range(batch_src_views):
                        rgb = rgbs[j]
                        if batch_tgt_src_mask[i, j].sum() > 0:
                            reprojected_xy = np.floor(xy[i, j][batch_tgt_src_mask[i, j]]).astype(np.int32) # shape of [Num_pts, 2] Note: a coarse design
                            rgb[list(reprojected_xy[:, 1]), list(reprojected_xy[:, 0])] = 0.3*rgb[list(reprojected_xy[:, 1]), list(reprojected_xy[:, 0])] + 0.7*self.mask_rgb
                        # imageio.imwrite(new_img_paths[idx+j], rgb)
                        save_rgbs.append(rgb)

            idx = idx + batch_src_views

            # torch.cuda.empty_cache()
        if save_dir is not None:
            video_dir = os.path.join(save_dir, 'reprojected_video.rgb.mp4')
            imageio.mimwrite(video_dir, np.stack(save_rgbs).astype(np.uint8), fps=30, quality=8)
            # np.save(os.path.join(save_dir, 'tgt2src_count.npy'), src_view_count.cpu().numpy())
            print("Write video into {}".format(video_dir))

        return src_view_count
    

    @torch.no_grad()
    def select_and_interp_views(self, valid_mask, tgt_pose, tgt_depth, tgt_intrinsic, \
                                src_pose_, src_depth, src_intrinsic, tgt_rgb=None, src_rgb=None, save_dir=None):
        '''
        Select training views for segmentation.

        Args:
        valid_mask: shape of [tgt_views, h, w]
        tgt_pose: shape of [tgt_views, 3, 4]
        src_pose: shape of [src_views, 3, 4]
        tgt_depth: shape of [tgt_views, h, w, 1]
        src_depth: shape of [src_views, h, w, 1]
        tgt_intrinsic: shape of [tgt_views, 3, 3]
        src_intrinsic: shape of [src_views, 3, 3]

        Return:
        src_view_count: shape of [tgt_views, src_views], representing how many pixels are projected into j^th src_view from i^th tgt_view.
        '''
        src_pose = src_pose_.detach().clone()
        if self.transfrom_matrix is not None:
            src_pose[:, :3, :3] = src_pose[:, :3, :3] @ self.transfrom_matrix

        src_view_count = self.reproject(tgt_pose, src_pose, tgt_depth, src_depth, \
                                        tgt_intrinsic, src_intrinsic, valid_mask=valid_mask, \
                                        tgt_rgb=tgt_rgb, src_rgb=src_rgb, save_dir=save_dir) # [1, src_views]
        idx_selected = src_view_count[0] > 10

        return idx_selected.cpu().numpy()




            
