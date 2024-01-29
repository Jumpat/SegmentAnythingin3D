import os
import numpy as np
import imageio

def cal_IoU(a, b):
    """Calculates the Intersection over Union (IoU) between two ndarrays.

    Args:
        a: shape (N, H, W).
        b: shape (N, H, W).

    Returns:
        Shape (N,) containing the IoU score between each pair of
        elements in a and b.
    """
    intersection = np.count_nonzero(np.logical_and(a == b, a != 0))
    union = np.count_nonzero(a + b)
    return intersection / union

def cal_IoU_from_path(a_path, b_path):
    """Calculates the Intersection over Union (IoU) between two images.

    Args:
        a_path: path to image a.
        b_path: path to image b.

    Returns:
        IoU score between image a and b.
    """
    a = imageio.imread(a_path) > 0
    b = imageio.imread(b_path) > 0
    return cal_IoU(a, b)

def cal_IoU_from_paths(a_paths, b_paths):
    """Calculates the Intersection over Union (IoU) between two sets of images.

    Args:
        a_paths: list of paths to images in set a.
        b_paths: list of paths to images in set b.

    Returns:
        Shape (N,) containing the IoU score between each pair of
        elements in a and b.
    """
    assert len(a_paths) == len(b_paths)
    a = np.stack([imageio.imread(path) > 0 for path in a_paths])
    b = np.stack([imageio.imread(path) > 0 for path in b_paths])
    return cal_IoU(a, b)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_path', type=str, required=True)
    parser.add_argument('--gt_path', type=str, required=True)
    args = parser.parse_args()

    if os.path.isdir(args.mask_path):
        assert(os.path.isdir(args.gt_path))
        mask_paths = [os.path.join(args.mask_path, f) for f in sorted(os.listdir(args.mask_path))]
        gt_paths = [os.path.join(args.gt_path, f) for f in sorted(os.listdir(args.gt_path))]
        iou = cal_IoU_from_paths(mask_paths, gt_paths)
    else:
        iou = cal_IoU_from_path(args.mask_path, args.gt_path)
    
    print('IoU: ', iou)