import numpy as np
import lib.EvaluationUtils as EvaluationUtils
from config import get_config
from lib.Evaluators import JMOD2Stats

config, unparsed = get_config()

#Edit model_name to choose model between ['jmod2','cadena','detector','depth','eigen']
model_name = 'jmod2'

model, detector_only = EvaluationUtils.load_model(model_name, config)

#Download the file from
dataset_file_path = "data/zurich_data/zurich_forest_dataset_with_obs_label.npy"
dataset = np.load(dataset_file_path).item()

len_data = len(dataset['images'])
showImages = True

#compute_depth_branch_stats_on_obs is set to False when evaluating detector-only models
stats = JMOD2Stats(model_name, compute_depth_branch_stats_on_obs=not detector_only)

for i in range(len_data):

    rgb = dataset['images'][i]
    gt = dataset['depth'][i]
    list_obstacles = dataset['obstacles'][i]

    results = model.run(rgb)

    corr_depth = None

    if results[0] is not None:
        depth_raw = results[0].copy()
        pred_depth_ = results[0][0, :, :, 0].copy()
        depth_gt = gt.copy()
        # evaluate only on valid predictions (some methods like Cadena's may return zero or negative values)
        depth_gt[np.nonzero(pred_depth_ <= 0)] = 0.0
        # evaluate only on valid GT depth(some pixels are invalid because of stereo matching failures)
        depth_gt[np.nonzero(depth_gt <= 0)] = 0.0
        pred_depth_[np.nonzero(depth_gt <= 0)] = 0.0
        # trim max depth to 39.75 (max depth the weights have been trained to estimate)
        depth_gt[np.nonzero(depth_gt > 39.75)] = 39.75
        results[0] = pred_depth_
    else:
        depth_gt = None
    #Corrected depth
    if results[2] is not None:
        corr_depth = results[2][0,:,:,0].copy()
        corr_depth[np.nonzero(depth_gt <= 0)] = 0.0
        results[2] = corr_depth

    gt_obs = EvaluationUtils.get_obstacles_from_list(list_obstacles)

    if showImages:
        if results[1] is not None:
            EvaluationUtils.show_detections(rgb, results[1], gt_obs, sleep_for=50)
        if results[0] is not None:
            EvaluationUtils.show_depth(rgb, depth_raw, gt, sleep_for=50)

    stats.run(results, [depth_gt, gt_obs])

results = stats.return_results()
