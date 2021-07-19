import sys
sys.path.append('../')
import numpy as np
import cv2
import math
import os
from ObstacleDetectionObjectives import numpy_iou

class Obstacle(object):
	def __init__(self, x, y, w, h, depth_seg = None, obs_stats = None):
		self.x = x #top left
		self.y = y #top left
		self.w = int(w)
		self.h = int(h)
		self.valid_points = -1
		if depth_seg is not None:
			self.segmentation = depth_seg[1]
			self.depth_mean, self.depth_variance = self.compute_depth_stats(depth_seg[0])
		elif obs_stats is not None:
			self.segmentation = None
			self.depth_mean = obs_stats[0]
			self.depth_variance = obs_stats[1]

	def compute_depth_stats(self, depth):
		if len(depth.shape) == 4:
			roi_depth = depth[0,self.y:self.y+self.h, self.x:self.x+self.w, 0]
		else:
			roi_depth = depth[self.y:self.y+self.h, self.x:self.x+self.w]

		if self.segmentation is not None:
			roi_segm = self.segmentation[self.y:self.y+self.h, self.x:self.x+self.w]

		mean_depth = 0
		squared_sum = 0
		valid_points = 0
		for y in range(0, self.h):
			for x in range(0, self.w):
				if self.segmentation is not None:
					if roi_segm[y,x] > 0 and roi_depth[y,x] > 0.:
						mean_depth += roi_depth.item(y,x)
						squared_sum += roi_depth.item(y,x)**2
						valid_points += 1
					else:
						if roi_depth[y,x] < 20.0 and roi_depth[y,x] > 0.0:
							mean_depth += roi_depth.item(y, x)
							squared_sum += roi_depth.item(y, x)**2
							valid_points += 1

		if self.valid_points > 0:
			mean_depth /= self.valid_points
			var_depth = (squared_sum - mean_depth**2) / self.valid_points
		elif valid_points > 0 and self.valid_points == -1:
			mean_depth /= valid_points
			var_depth = (squared_sum - mean_depth ** 2) / valid_points
			self.valid_points = valid_points
		else:
			mean_depth = -1
			var_depth = -1
		return mean_depth, var_depth
	def evaluate_estimation(self, estimated_depth):
		mean_rmse = (self.depth_mean - self.depth_mean)**2
		mean_variance = (self.depth_variance - self.depth_variance)**2
		return math.sqrt(mean_rmse), math.sqrt(mean_variance), self.valid_points

def depth_to_meters_airsim(depth):
	depth = depth.astype(np.float64)
	for i in range(0, depth.shape[0]):
		for j in range(0, depth.shape[1]):
			depth[i,j] = (-4.586e-09 * (depth[i,j] ** 4.)) + (3.382e-06 * (depth[i,j] ** 3.)) - (0.000105 * (depth[i,j] ** 2.)) + (0.04239 * depth[i,j]) + 0.04072
	return depth

def depth_to_meters_base(depth):
	return depth * 39.75 / 255.

def get_obstacles_from_list(list):
	obstacles = []
	for obstacle_def in list:
		obstacle = Obstacle(obstacle_def[0][0], obstacle_def[0][1], obstacle_def[0][2], obstacle_def[0][3], obs_stats=(obstacle_def[1][0], obstacle_def[1][1]))
		obstacles.append(obstacle)
	return obstacles

def get_detected_obstacles_from_detector(prediction, confidence_thr = 1.0, output_img = None):
	def sigmoid(x):
		return 1 / (1 + math.exp(-x))

	if len(prediction.shape) == 4:
		prediction = np.expand_dims(prediction, axis=0)

	confidence = []
	conf_pred = prediction[0, :, :, :, 0]
	x_pred = prediction[0, :, :, :, 1]
	y_pred = prediction[0, :, :, :, 2]
	w_pred = prediction[0, :, :, :, 3]
	h_pred = prediction[0, :, :, :, 4]
	mean_pred = prediction[0, :, :, :, 5]
	var_pred = prediction[0, :, :, :, 6]
	
	# img shape
	IMG_WIDTH = 256
	IMG_HEIGHT = 160

	# Anchors
	anchors = np.array([[0.34755122, 0.84069513], 
						[0.14585618, 0.25650666]], dtype=np.float32)

	# obstacles list
	file = open('confidence.txt', 'a')
	detected_obstacles = []
	for i in range(0, 5):
		for j in range(0, 8):
			for k in range(0, 2):
				val_conf = sigmoid(conf_pred[i, j, k])
				file.write(str(sigmoid(conf_pred[i, j, k]))+'\n')
				if val_conf >= confidence_thr:
					x = sigmoid(x_pred[i, j, k])
					y = sigmoid(y_pred[i, j, k])
					w = np.exp(w_pred[i, j, k]) * anchors[k, 0] * IMG_WIDTH
					h = np.exp(h_pred[i, j, k]) * anchors[k, 1] * IMG_HEIGHT
					mean = mean_pred[i, j, k] * 39.75
					var = var_pred[i, j, k] * 39.75 * 100
					x_top_left = int(np.floor(((x + j) * 32.) - (w / 2.)))
					y_top_left = int(np.floor(((y + i) * 32.) - (h / 2.)))
					if output_img is not None:
						cv2.rectangle(output_img, (x_top_left, y_top_left), (x_top_left+int(w), y_top_left+int(h)), (0,0,255), 2)
					detected_obstacles.append([(x_top_left, y_top_left, w, h), (mean, var)])
	obstacles = get_obstacles_from_list(detected_obstacles)
	file.close()
	return obstacles, output_img

def get_obstacles_from_seg_and_depth(depth, segm, depth_thr = 20, segm_thr = 55, f_segm_thr = cv2.THRESH_BINARY, is_gt = False):
	"Given segmentation and depth, get a list of obstacles objects. Depth_thr: max distance of obstacles Segm_thr= threshold between obstacle/non-obstacle classes"
	if len(depth.shape) == 4:
		depth = depth[0,:,:,0]
	if len(segm.shape) == 4:
		segm = segm[0,:,:,0]

	retval_d, depth_mask = cv2.threshold(depth, depth_thr, 1., cv2.THRESH_BINARY_INV)
	if segm_thr == -1:
		#segmentazione binaria fornita. espando a 255
		retval, obstacles_mask = cv2.threshold(segm*255, 127, 1., cv2.THRESH_BINARY)
	else:
		# threshold segmentation
		retval, obstacles_mask = cv2.threshold(segm, int(segm_thr), 1., f_segm_thr)
	# remove contours for thresholded segmentation
	obstacles_mask = obstacles_mask.astype(np.float32) * depth_mask

	im2, contours, hierarchy = cv2.findContours(obstacles_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contours_img = np.zeros(shape=obstacles_mask.shape)
	contours_img = cv2.drawContours(contours_img, contours, -1, (1, 1, 1), 2)
	obstacles_mask = (obstacles_mask) - contours_img.astype(np.uint8)
	obstacles = []

	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		obstacle = Obstacle(x, y, w, h, depth_seg=(depth_mask * depth * (obstacles_mask), obstacles_mask))
		if obstacle.valid_points > 35:
			obstacles.append(obstacle)

		return obstacles

def rmse_error_on_vector(y_true, y_pred):
	diff = y_true - y_pred
	square = np.square(diff)
	# mean error
	mean = np.sum(square) / float(np.count_nonzero(y_true))
	rmse_error = np.sqrt(mean + 0.00001)
	return rmse_error

def sc_inv_logrmse_error_on_vector(y_true, y_pred):
	first_log = np.log(y_pred + 1.)
	second_log = np.log(y_true + 1.)

	log_term = np.sum(np.square((first_log - second_log))) / float(np.count_nonzero(first_log))
	sc_inv_term = np.square(np.sum(first_log - second_log) / (float(np.count_nonzero(first_log))**2))
	error = log_term - sc_inv_term
	return error

def rmse_error_on_matrix(y_true, y_pred):
	y_true = y_true.flatten()
	y_pred = y_pred.flatten()
	rmse_error = rmse_error_on_vector(y_true,y_pred)
	return rmse_error

def sc_inv_logrmse_error_on_matrix(y_true, y_pred):
	y_true = y_true.flatten()
	y_pred = y_pred.flatten()
	error = sc_inv_logrmse_error_on_vector(y_true, y_pred)
	return error

def compute_obstacle_error_on_depth_branch(estimation, obstacles, output_img = None):
	"Given a depth estimation and a list of obstacles, compute depth error on obstacles"
	i = 0
	obs_area = 0
	obs_m_error = 0
	obs_v_error = 0

	for obstacle in obstacles:
		m_error, v_error, valid_points = obstacle.evaluate_estimation(estimation)
		area = valid_points  # w * h
		if (m_error != -1) and valid_points > 35: #arbitrary threshold for small obstacles
			obs_area += area
			obs_m_error += m_error*area
			obs_v_error += v_error*area

		if output_img is not None:
			error_text = ("%.2f,%.2f" %(obstacle.depth_mean, m_error))
			cv2.rectangle(output_img, (obstacle.x, obstacle.y), (obstacle.x + obstacle.w, obstacle.y + obstacle.h), (0, 255, 0), 2)
		i += 1

	if (obs_m_error is 0 and obs_area is 0):
		obs_m_error = -1
		obs_v_error = -1

	return obs_m_error, obs_v_error, obs_area, output_img

def compute_detection_stats(detected_obstacles, gt_obstacles, iou_thresh = 0.25):
	#convert in Obstacle object the input list, created by get_detected_obstacles_from_detector
	#print "Detected {} obstacles. The image has {} GT obstacles".format(len(detected_obstacles), len(gt_obstacles))
	if len(gt_obstacles) > 0:
		closer_gt_obstacles = []

		for det_obstacle in detected_obstacles:
			#Find in GT closer obstacle to the one detected
			max_idx = 0
			idx = 0
			max_iou = 0
			is_overlap = 0

			#Uso IOU per questa misura. E se misurassi distanza dei centri?
			for gt_obstacle in gt_obstacles:
				iou, overlap = numpy_iou((gt_obstacle.x + gt_obstacle.w / 2, gt_obstacle.y + gt_obstacle.h / 2),
										 (det_obstacle.x + det_obstacle.w / 2, det_obstacle.y + det_obstacle.h / 2),
										 (gt_obstacle.w, gt_obstacle.h),
										 (det_obstacle.w, det_obstacle.h))
				if iou > max_iou:
					max_iou = iou
					max_idx = idx
					is_overlap = overlap #if one of the obstacles is contained in another

				idx += 1
			closer_gt_obstacles.append((gt_obstacles[max_idx], max_idx, max_iou, is_overlap))
		
		# Resultado
		iou_for_each_gt_obstacle = np.zeros(shape=len(gt_obstacles), dtype=np.float32)
		depth_error_for_each_gt_obstacle = np.zeros(shape=len(gt_obstacles), dtype=np.float32)
		var_depth_error_for_each_gt_obstacle = np.zeros(shape=len(gt_obstacles), dtype=np.float32)
		n_valid_pred_for_each_gt_obstacle = np.zeros(shape=len(gt_obstacles))

		it = 0

		for elem in closer_gt_obstacles:
			if elem[2] > iou_thresh:
				n_valid_pred_for_each_gt_obstacle[elem[1]] += 1
				if elem[2] > iou_for_each_gt_obstacle[elem[1]]:
					iou_for_each_gt_obstacle[elem[1]] = elem[2]
					depth_error_for_each_gt_obstacle[elem[1]] = rmse_error_on_vector(elem[0].depth_mean, detected_obstacles[it].depth_mean)
					var_depth_error_for_each_gt_obstacle[elem[1]] = rmse_error_on_vector(elem[0].depth_variance, detected_obstacles[it].depth_variance)
				it += 1
		
		n_detected_obstacles = 0
		n_non_detected_obs = 0
		
		for n in n_valid_pred_for_each_gt_obstacle:
			if n > 0:
				n_detected_obstacles += 1
			else:
				n_non_detected_obs += 1

		avg_iou = 0
		avg_mean_depth_error = -1
		avg_var_depth_error = -1

		if n_detected_obstacles > 0:
			avg_iou = np.mean(iou_for_each_gt_obstacle[np.nonzero(iou_for_each_gt_obstacle)])
			avg_mean_depth_error = np.mean(depth_error_for_each_gt_obstacle[np.nonzero(depth_error_for_each_gt_obstacle)])
			avg_var_depth_error = np.mean(var_depth_error_for_each_gt_obstacle[np.nonzero(var_depth_error_for_each_gt_obstacle)])
		
		true_positives = np.sum(n_valid_pred_for_each_gt_obstacle)
		false_positives = len(detected_obstacles) - true_positives
		multiple_detections = true_positives - n_detected_obstacles
		precision = true_positives / (true_positives + false_positives)
		recall = true_positives / (true_positives + n_non_detected_obs)
	
	elif len(detected_obstacles) > 0:
		#detection on image with no gt obstacle
		avg_iou = 0
		precision = 0
		recall= 0
		avg_mean_depth_error = -1
		avg_var_depth_error = -1
		true_positives = 0
		false_positives = len(detected_obstacles)
		n_non_detected_obs = 0
	else:
		# detection on image with no gt obstacle
		avg_iou = -1
		precision = -1
		recall = -1
		avg_mean_depth_error = -1
		avg_var_depth_error = -1
		true_positives = -1
		false_positives = -1
		n_non_detected_obs = -1

	return avg_iou, precision, recall, avg_mean_depth_error, avg_var_depth_error, true_positives, false_positives, n_non_detected_obs

def show_detections(rgb, detection, gt=None, save = True, save_dir = 'result', file_name=None, print_depths=False, sleep_for = 50):
	if len(rgb.shape) == 4:
		rgb = rgb[0, :, :, :]

	if len(rgb.shape) == 3 and rgb.shape[2] == 1:
		rgb = rgb[:,:,0]

	if len(rgb.shape) == 2:
		rgb_new = np.zeros(shape=(rgb.shape[0],rgb.shape[1],3))
		rgb_new[:,:,0] = rgb
		rgb_new[:,:,1] = rgb
		rgb_new[:,:,2] = rgb
		rgb = rgb_new
	
	output = rgb.copy()
	det_obstacles_data = []
	gt_obstacles_data = []

	for obs in detection:
		cv2.rectangle(output,(int(obs.x),int(obs.y)),(int(obs.x) + int(obs.w), int(obs.y)+int(obs.h)),(0,0,255),2)
		det_obstacles_data.append((obs.x, obs.y, obs.w, obs.h, obs.depth_mean, obs.depth_variance))

	if gt is not None:
		for obs in gt:
			cv2.rectangle(output, (int(obs.x),int(obs.y)),(int(obs.x) + int(obs.w), int(obs.y)+int(obs.h)), (0, 255, 0), 2)
			gt_obstacles_data.append((obs.x, obs.y, obs.w, obs.h, obs.depth_mean, obs.depth_variance))

	if save:
		abs_save_dir = os.path.join(os.getcwd(),save_dir)
		if not os.path.exists(os.path.join(abs_save_dir,'rgb')):
			os.makedirs(os.path.join(abs_save_dir,'rgb'))
		if not os.path.exists(os.path.join(abs_save_dir,'detections')):
			os.makedirs(os.path.join(abs_save_dir, 'detections'))

		cv2.imwrite(os.path.join(abs_save_dir,'rgb',file_name), rgb)
		cv2.imwrite(os.path.join(abs_save_dir,'detections', file_name), output)

		with open(os.path.join(abs_save_dir,'detections', os.path.splitext(file_name)[0] + '.txt'),'w') as f:
			f.write('Detected obstacles\n')
			for x in det_obstacles_data:
				f.write('x:{},y:{},w:{},h:{},depth:{},var_depth:{}\n'.format(x[0], x[1], x[2], x[3], x[4], x[5]))
			if gt is not None:
				f.write('\nGT obstacles\n')
				for x in gt_obstacles_data:
					f.write('x:{},y:{},w:{},h:{},depth:{},var_depth:{}\n'.format(x[0], x[1], x[2], x[3], x[4], x[5]))
	
	cv2.imshow("Detections(RED:predictions,GREEN: GT", output)
	cv2.waitKey(sleep_for)

def show_depth(rgb, depth, gt=None, save = True, save_dir = 'result', file_name=None, max = 45.0, sleep_for=50):
	if len(rgb.shape) == 4:
		rgb = rgb[0, :, :, :]
	if len(depth.shape) == 4:
		depth = depth[0, :, :, :]
	if len(gt.shape) == 4:
		gt = gt[0,...]

	depth_img = np.clip(depth[:, :], 0.0, max)
	depth_img = (depth_img/max * 255.).astype("uint8")
	depth_jet = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)

	cv2.imshow("Predicted Depth", depth_jet)

	if gt is not None:
		gt_img = np.clip(gt, 0.0, max)
		gt_img = (gt/max * 255.).astype("uint8")
		gt_jet = cv2.applyColorMap(gt_img, cv2.COLORMAP_JET)
		cv2.imshow("GT Depth", gt_jet)

	if save:
		abs_save_dir = os.path.join(os.getcwd(), save_dir)
		if not os.path.exists(os.path.join(abs_save_dir,'rgb')):
			os.makedirs(os.path.join(abs_save_dir,'rgb'))
		if not os.path.exists(os.path.join(abs_save_dir, 'depth')):
			os.makedirs(os.path.join(abs_save_dir, 'depth'))
		if gt is not None:
			if not os.path.exists(os.path.join(abs_save_dir, 'gt')):
				os.makedirs(os.path.join(abs_save_dir, 'gt'))
			cv2.imwrite(os.path.join(abs_save_dir, 'gt', file_name), gt_jet)
		
		cv2.imwrite(os.path.join(abs_save_dir, 'rgb', file_name), rgb)
		cv2.imwrite(os.path.join(abs_save_dir, 'depth', file_name), depth_jet)

	cv2.waitKey(sleep_for)


def load_model(name, config):
	from models.JMOD2 import JMOD2
	
	model = JMOD2(config)
	model.model.load_weights("weights/jmod2.hdf5")
	detector_only = False

	return model, detector_only