#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, fetchPly
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    # target: seg, scene
    def __init__(self, args : ModelParams, gaussians : GaussianModel=None, load_iteration=None, shuffle=True, resolution_scales=[1.0], init_from_3dgs_pcd=False, target='scene', mode='train'):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                if mode == 'train':
                    if target == 'seg' or target == 'coarse_seg_everything':
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
                    elif target == 'scene':
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
                    else:
                        assert False and "Unknown target!"
                elif mode == 'eval':
                    if target == 'seg':
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="seg")
                    elif target == 'scene':
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
                    else:
                        assert False and "Unknown target!"
            else:
                self.loaded_iter = load_iteration

            print("Loading trained model at iteration {}".format(self.loaded_iter))
            
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, need_features = args.need_features, need_masks = False, replica = 'replica' in args.model_path)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # Load or initialize scene / seg gaussians
        if self.loaded_iter and self.gaussians is not None:
            if mode == 'train':
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "scene_point_cloud.ply"))
            else:
                if target == 'coarse_seg_everything':
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "scene_point_cloud.ply"))
                else:
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            target+"_point_cloud.ply"))
        elif self.gaussians is not None:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)


    def save(self, iteration, target='scene', colored = False):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        if not colored:
            self.gaussians.save_ply(os.path.join(point_cloud_path, target+"_point_cloud.ply"))
        else:
            print("Warning: Colored point cloud does not preserve the Gaussian parameters!")
            self.gaussians.save_colored_ply(os.path.join(point_cloud_path, target+"_point_cloud_colored.ply"))
    
    def save_mask(self, iteration, id = 0):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_mask(os.path.join(point_cloud_path, f"seg_point_cloud_{id}.npy"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]