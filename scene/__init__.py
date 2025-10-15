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
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, cameraList_from_camInfos_parallel, camera_to_JSON
from torch import nn
from rich.console import Console
import torch
CONSOLE = Console(width=120)
class Scene:

    gaussians: GaussianModel
    def __init__(self, args : ModelParams, gaussians : GaussianModel, shuffle=True, resolution_scales=[1.0], dynamic_training = False, load_frame_id = -1, ply_path = None,
                  parallel_load = False, no_image=False, stage = 1, warpDQB=None, seq=False, key_frame=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.stage = stage
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.dynamic_training = dynamic_training

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, load_frame_id = load_frame_id, ply_path = ply_path, no_image=no_image)
        elif os.path.exists(os.path.join(args.source_path, "sparse")) or os.path.exists(os.path.join(args.source_path, "colmap", "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, load_frame_id = load_frame_id, ply_path = ply_path)
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
            if not parallel_load:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            else:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos_parallel(scene_info.train_cameras, resolution_scale, args, no_image=no_image)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos_parallel(scene_info.test_cameras, resolution_scale, args, no_image=no_image)
    
        if self.dynamic_training:
            try:
                if self.stage == 2:
                    CONSOLE.log("loading motion from control point")
                    # 檢查是否需要初始化 skin2Joint interpolation
                    # 對於 key frame 模式，需要在第一個非起始幀時初始化
                    if (load_frame_id == warpDQB.stFrame_ + warpDQB.step_ or
                        (key_frame is not None and hasattr(warpDQB, 'raw_xyz') and
                         not hasattr(warpDQB, 'indices_'))):
                        CONSOLE.log(f"Initializing skin2Joint interpolation for frame {load_frame_id}")
                        warpDQB.skin2JointInterpolation(warpDQB.raw_xyz)

                    if seq:
                        # 對於 sequential 模式，判斷訓練方向決定參考幀
                        if key_frame is not None and load_frame_id < key_frame:
                            # Backward training: 載入後一幀（已訓練過的）
                            reference_frame = load_frame_id + warpDQB.step_
                            CONSOLE.log(f"Backward training: loading frame {reference_frame} for frame {load_frame_id}")
                        else:
                            # Forward training: 載入前一幀
                            reference_frame = load_frame_id - warpDQB.step_
                        
                        self.gaussians.load_ply(os.path.join(self.model_path,
                                                                    "ckt",
                                                                    "point_cloud_%d.ply") % reference_frame, self.cameras_extent)
                        warpDQB.warping(self.gaussians, self.gaussians._xyz, self.gaussians._rotation)
                    else:
                        # 對於非 sequential 模式，優先嘗試從 key_frame 載入
                        if key_frame is not None and key_frame != load_frame_id:
                            try:
                                CONSOLE.log(f"Trying to load from key frame: {key_frame}")
                                self.gaussians.load_ply(os.path.join(self.model_path,
                                                                            "ckt",
                                                                            "point_cloud_%d.ply") % key_frame, self.cameras_extent)
                            except:
                                CONSOLE.log("Key frame not found, falling back to start frame")
                                self.gaussians.load_ply(os.path.join(self.model_path,
                                                                            "ckt",
                                                                            "point_cloud_%d.ply") % (warpDQB.stFrame_), self.cameras_extent)
                        else:
                            self.gaussians.load_ply(os.path.join(self.model_path,
                                                                        "ckt",
                                                                        "point_cloud_%d.ply") % (warpDQB.stFrame_), self.cameras_extent)
                        warpDQB.warping(self.gaussians, warpDQB.raw_xyz, warpDQB.raw_rot)

                else:
                    # Stage 1: 判斷訓練方向決定參考幀
                    if key_frame is not None and load_frame_id < key_frame:
                        # Backward training: 載入後一幀（已訓練過的）
                        reference_frame = load_frame_id + warpDQB.step_
                        CONSOLE.log(f"Stage 1 Backward training: loading frame {reference_frame} for frame {load_frame_id}")
                    else:
                        # Forward training: 載入前一幀
                        reference_frame = load_frame_id - warpDQB.step_
                    
                    try:
                        self.gaussians.load_ply(os.path.join(self.model_path,
                                                                    "ckt",
                                                                    "point_cloud_%d.ply") % reference_frame, self.cameras_extent)
                    except:
                        if key_frame is not None and key_frame != load_frame_id:
                            CONSOLE.log(f"Reference frame {reference_frame} not found, trying key frame: {key_frame}")
                            self.gaussians.load_ply(os.path.join(self.model_path,
                                                                        "ckt",
                                                                        "point_cloud_%d.ply") % key_frame, self.cameras_extent)
                        else:
                            raise  # 重新拋出異常，讓下面的 except 處理
                    
                    with torch.no_grad():
                        self.gaussians._xyz += warpDQB.xyz_velocity
               
            except:
                CONSOLE.log("Could not load dynamic_training point cloud, creating from scratch")
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]