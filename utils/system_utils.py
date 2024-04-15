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

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

# def searchForMaxIteration(folder):
#     saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
#     return max(saved_iters)

# target: feature, seg, scene
def searchForMaxIteration(folder, target = "scene"):
    fnames = os.listdir(folder)
    saved_iters = []
    for fname in fnames:
        cur_dir = os.path.join(folder, fname)
        plys = os.listdir(cur_dir)
        has_target_ply = False
        for p in plys:
            if target in p:
                has_target_ply = True
                break
        if has_target_ply:
            saved_iters.append(int(fname.split("_")[-1]))
    try:
        return max(saved_iters)
    except:
        return None
