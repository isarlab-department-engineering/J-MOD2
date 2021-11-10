import DepthMetrics
from EvaluationUtils import *

class Stats(object):
	def __init__(self, string_id):
		self.string_id =  string_id
		self.iterations = 0

	def measure_performance(self, prediction, gt):
		print "implement"

	def run(self, prediction, gt):
		self.iterations += 1
		self.measure_performance(prediction, gt)
		self.print_stats()

	def print_stats(self):
		print "******************"
		self.print_stats_spec()

	def return_results(self):
		#return a dictionary with results
		raise NotImplementedError


class DepthStats(Stats):
	def __init__(self, string_id):
		self.rmse_depth_acc = 0
		self.sc_inv_depth_acc = 0
		super(DepthStats, self).__init__(string_id)

	def measure_performance(self, prediction, gt):
		if len(prediction.shape) > 1:
			if len(gt.shape) == 4:
				gt = gt[0, :, :, 0]
			elif len(gt.shape) == 3:
				gt = gt[:, :, 0]
			if len(prediction.shape) == 4:
				prediction = prediction[0, :, :, 0]
			elif len(prediction.shape) == 3:
				prediction = prediction[:, :, 0]
			self.rmse_depth_acc += rmse_error_on_matrix(gt, prediction)
			self.sc_inv_depth_acc += sc_inv_logrmse_error_on_matrix(gt, prediction)
		elif len(prediction.shape) == 1 and len(gt.shape) == 1:
			self.rmse_depth_acc += rmse_error_on_vector(gt,prediction)
			self.sc_inv_depth_acc += sc_inv_logrmse_error_on_vector(gt, prediction)
		else:
			"INCONSISTENCY: DEPTHSTATS GT shape:{} PRED shape: {}".format(gt.shape, prediction.shape)

	def print_stats_spec(self):
		print("{} Depth RMSE/Sc.Inv: {},{}").format(self.string_id, self.rmse_depth_acc / float(self.iterations),
																	self.sc_inv_depth_acc / float(self.iterations))
	
	def return_results(self):
		errors = {}
		errors['name_experiment'] = self.string_id
		errors['n_evaluations'] = self.iterations
		errors['RMSE'] = self.rmse_depth_acc / float(self.iterations + np.finfo(float).eps)
		errors['Scale_Inv_MSE'] = self.sc_inv_depth_acc / float(self.iterations + np.finfo(float).eps)
		return errors


class DetectionStats(Stats):
	def __init__(self, string_id, iou_thresh = 0.5):
		self.iou_acc = 0
		self.true_pos_acc = 0
		self.false_pos_acc = 0.00000001
		self.false_neg_acc = 0.00000001
		self.valid_detections = 0.000000001
		self.imgs_with_obs = 0.0000000001
		self.depth_m = 0
		self.depth_v = 0
		self.iou_thresh = iou_thresh
		super(DetectionStats, self).__init__(string_id)

	def measure_performance(self, pred_obstacles, gt):
		iou, precision, recall, depth_m, depth_v, t, f, n = compute_detection_stats(pred_obstacles, gt, iou_thresh=0.5)

		if iou > -1:
			self.imgs_with_obs += 1
			self.iou_acc += iou
			self.true_pos_acc += t
			self.false_pos_acc += f
			self.false_neg_acc += n
		if depth_m > -1:
			self.valid_detections += 1
			self.depth_m += depth_m
			self.depth_v += depth_v

	def print_stats_spec(self):
		print("{} Detector IOU/Precision/Recall: {}, {}, {}").format(self.string_id, self.iou_acc/self.imgs_with_obs,
																	 self.true_pos_acc / (self.true_pos_acc + self.false_pos_acc),
																	 self.true_pos_acc / (self.true_pos_acc + self.false_neg_acc))
		print("{} Detector Mean/Variance: {}, {}").format(self.string_id, self.depth_m / self.valid_detections,
														  self.depth_v / self.valid_detections)

	def return_results(self):
		errors = {}
		errors['name_experiment'] = self.string_id
		errors['n_evaluations'] = self.iterations
		errors['valid_detections'] = self.valid_detections
		errors['IOU'] = self.iou_acc / self.imgs_with_obs
		errors['Precision'] = self.true_pos_acc / (self.true_pos_acc + self.false_pos_acc)
		errors['Recall'] = self.true_pos_acc / (self.true_pos_acc + self.false_neg_acc)
		errors['Detector_Depth_Mean_Error'] = self.depth_m / self.valid_detections
		errors['Detector_Depth_Var_Error'] = self.depth_v / self.valid_detections
		return errors


class DepthOnObstacles(Stats):
	def __init__(self, string_id):
		self.obs_area_acc = 0.0000001
		self.depth_error_on_obs_acc = 0
		self.depth_var_error_on_obs_acc = 0
		super(DepthOnObstacles, self).__init__(string_id)

	def measure_performance(self, prediction, gt):
		obs_m_err, obs_v_err, obs_area, show_img = compute_obstacle_error_on_depth_branch(prediction, gt)
		if obs_m_err > -1:
			self.obs_area_acc += obs_area
			self.depth_error_on_obs_acc += obs_m_err
			self.depth_var_error_on_obs_acc += obs_v_err

	def print_stats_spec(self):
		print("{} Depth Error On Obst:{} (var: {})").format(self.string_id, self.depth_error_on_obs_acc / float(self.obs_area_acc),
															self.depth_var_error_on_obs_acc  / float(self.obs_area_acc))

	def return_results(self):
		errors = {}
		errors['name_experiment'] = self.string_id
		errors['n_evaluations'] = self.iterations
		errors['Depth_Error_on_Obs'] = self.depth_error_on_obs_acc / float(self.obs_area_acc)
		errors['Depth Var Error on Obs'] = self.depth_var_error_on_obs_acc  / float(self.obs_area_acc)
		return errors


class JMOD2Stats(Stats):
	def __init__(self, string_id, compute_depth_branch_stats_on_obs, iou_thresh=0.5):
		self.depth_stats = DepthStats(string_id + " Depth")
		self.corrected_depth_stats = DepthStats(string_id + "Corrected Depth Branch")
		self.depth_stats_on_obstacles = DepthOnObstacles(string_id +" Non-corrected Depth Error on Obstacles")
		self.corr_depth_stats_on_obstacles = DepthOnObstacles(string_id + " Corrected Depth Error on Obstacles")
		self.detection_stats = DetectionStats(string_id + " Detection Branch", iou_thresh=iou_thresh)
		self.compute_depth_branch_stats_on_obs = compute_depth_branch_stats_on_obs
		super(JMOD2Stats, self).__init__(string_id)

	def measure_performance(self, prediction, gt):
		#prediction: [depth, detection(raw), corrected_depth] gt: [depth, obstacles(processed)
		if prediction[0] is not None and gt[0] is not None:
			self.depth_stats.run(prediction[0], gt[0])
		if prediction[2] is not None and gt[0] is not None:
			self.corrected_depth_stats.run(prediction[2], gt[0])
		if prediction[1] is not None and gt[1] is not None:
			self.detection_stats.run(prediction[1], gt[1])
		if self.compute_depth_branch_stats_on_obs and gt[1] is not None:
			self.depth_stats_on_obstacles.run(prediction[0], gt[1])
			if prediction[2] is not None:
				self.corr_depth_stats_on_obstacles.run(prediction[2], gt[1])
	
	def print_stats(self):
		print "/////////////////////////////////"
		#logs are printed by submodules

	def return_results(self):
		errors = {}
		errors['name_experiment'] = self.string_id
		errors['depth_stats'] = self.depth_stats.return_results()
		errors['corrected_depth_stats'] = self.corrected_depth_stats.return_results()
		errors['detection_stats'] = self.detection_stats.return_results()
		errors['depth_error_on_obs'] = self.depth_stats_on_obstacles.return_results()
		errors['corr_depth_error_on_obs'] = self.corr_depth_stats_on_obstacles.return_results()
		return errors