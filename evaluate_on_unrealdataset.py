import cv2
import numpy as np
import lib.EvaluationUtils as EvaluationUtils
from config import get_config
import os
from glob import glob
from lib.Evaluators import JMOD2Stats


def preprocess_data(rgb, gt, seg, w=256, h=160, crop_w=0, crop_h=0, resize_only_rgb = False):
    crop_top = np.floor((rgb.shape[0] - crop_h) / 2).astype(np.uint8)
    crop_bottom = rgb.shape[0] - np.floor((rgb.shape[0] - crop_h) / 2).astype(np.uint8)
    crop_left = np.floor((rgb.shape[1] - crop_w) / 2).astype(np.uint8)
    crop_right = rgb.shape[1] - np.floor((rgb.shape[1] - crop_w) / 2).astype(np.uint8)

    rgb = np.asarray(rgb, dtype=np.float32) / 255.
    rgb = cv2.resize(rgb, (w, h), cv2.INTER_LINEAR)
    rgb = np.expand_dims(rgb, 0)
    gt = np.asarray(gt, dtype=np.float32)

    if not resize_only_rgb:
        gt = cv2.resize(gt, (w, h), cv2.INTER_NEAREST)
    gt = EvaluationUtils.depth_to_meters_airsim(gt)
    if not resize_only_rgb:
        seg = cv2.resize(seg, (w, h), cv2.INTER_NEAREST)
    return rgb, gt, seg

#edit config.py as required
config, unparsed = get_config()

#Edit model_name to choose model between ['jmod2','cadena','detector','depth','eigen']
model_name = 'jmod2'

model, detector_only = EvaluationUtils.load_model(model_name, config)

showImages = False

dataset_main_dir = config.data_set_dir
test_dirs = config.data_test_dirs

#compute_depth_branch_stats_on_obs is set to False when evaluating detector-only models
jmod2_stats = JMOD2Stats(model_name, compute_depth_branch_stats_on_obs=not detector_only)


for test_dir in test_dirs:
    depth_gt_paths = sorted(glob(os.path.join(dataset_main_dir, test_dir, 'depth', '*' + '.png')))
    rgb_paths = sorted(glob(os.path.join(dataset_main_dir, test_dir, 'rgb', '*' + '.png')))
    seg_paths = sorted(glob(os.path.join(dataset_main_dir, test_dir, 'segmentation', '*' + '.png')))


    for gt_path, rgb_path, seg_path in zip(depth_gt_paths, rgb_paths, seg_paths):

        rgb_raw = cv2.imread(rgb_path)
        gt = cv2.imread(gt_path, 0)
        seg = cv2.imread(seg_path, 0)

        #Normalize input between 0 and 1, resize if required

        rgb, gt, seg = preprocess_data(rgb_raw, gt, seg, w=config.input_width,h=config.input_height, resize_only_rgb = True)

        #Forward pass to the net
        results = model.run(rgb)

        if results[0] is not None:
            depth_raw = results[0].copy()
            pred_depth_ = results[0][0, :, :, 0].copy()
            depth_gt = gt.copy()
            #evaluate only on valid predictions (some methods like Cadena's may return zero or negative values)
            depth_gt[np.nonzero(pred_depth_ <= 0)] = 0.0
        else:
            depth_gt = None
        #Corrected depth
        if results[2] is not None:
            corr_depth = results[2][0, :, :, 0].copy()
            corr_depth[np.nonzero(depth_gt <= 0)] = 0.0
            results[2] = corr_depth

        #Get obstacles from GT segmentation and depth
        gt_obs = EvaluationUtils.get_obstacles_from_seg_and_depth(gt, seg, segm_thr=-1)

        if showImages:
            if results[1] is not None:
                EvaluationUtils.show_detections(rgb_raw, results[1], gt_obs, sleep_for=10)
            if results[0] is not None:
                EvaluationUtils.show_depth(rgb_raw, depth_raw, gt,sleep_for=10)

        jmod2_stats.run(results, [depth_gt, gt_obs])

results = jmod2_stats.return_results()
