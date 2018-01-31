import lib.EvaluationUtils as EvaluationUtils
import h5py
import numpy as np
from lib.Evaluators import JMOD2Stats

from config import get_config


config, unparsed = get_config()
#Edit model_name to choose model between ['jmod2','cadena','detector','depth','eigen']
model_name = 'jmod2'

model, detector_only= EvaluationUtils.load_model(model_name, config)
#Download and zip the dataset from
main_data_dir = "data/zurich_data"
hdf5_file = main_data_dir + "/seq_%02d.h5"

showImages = True
#compute_depth_branch_stats_on_obs is set to False when evaluating detector-only models
stats = JMOD2Stats(model_name, compute_depth_branch_stats_on_obs=not detector_only)

for k in range(0,4):
    zurich_data = h5py.File(hdf5_file%k,'r')

    images_t = np.transpose(np.asarray(zurich_data['data'], dtype=np.float32)[:,:,:,:],[0,2,3,1])
    images = np.zeros(shape=(images_t.shape[0],images_t.shape[1],images_t.shape[2],3))
    #gray to RGB
    images[:,:,:,0] = images_t[:,:,:,0]
    images[:,:,:,1] = images_t[:,:,:,0]
    images[:,:,:,2] = images_t[:,:,:,0]

    gt = np.transpose(np.asarray(zurich_data['label'], dtype=np.float32),[0,2,3,1])

    for i in range(0,images.shape[0]):
        img = images[i,:,:,:]
        gt_ = gt[i,:,:,0]
        results = model.run(img)
        corr_depth = None
        if results[0] is not None:
            depth_raw = results[0].copy() #to be used only by show_depth function
            pred_depth_ = results[0][0, :, :, 0].copy()
            depth_gt = gt_.copy()
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
        # Corrected depth
        if results[2] is not None:
            corr_depth = results[2][0, :, :, 0].copy()
            corr_depth[np.nonzero(depth_gt <= 0)] = 0.0
            results[2] = corr_depth

        if showImages:
            if results[1] is not None:
                EvaluationUtils.show_detections(img, results[1], gt=None, sleep_for=10)
            if results[0] is not None:
                EvaluationUtils.show_depth(img, depth_raw, gt[i,:,:,0], sleep_for=10)

        stats.run(results, [depth_gt, None])

    del images
    zurich_data.close()
    del images_t

results = stats.return_results()

