import numpy as np
import cv2

class Sample(object):
	def __init__(self, img_path, label_path):
		self.img_path = img_path
		self.depth_gt = label_path[0]
		self.obstacles_gt = label_path[1]

	def read_features(self):
		img = cv2.imread(self.img_path[0], cv2.IMREAD_COLOR)
		return img

	def numpy_overlap(self, x1, w1, x2, w2):
		l1 = (x1) - w1 / 2
		l2 = (x2) - w2 / 2
		left = np.where(l1 > l2, l1, l2)
		r1 = (x1) + w1 / 2
		r2 = (x2) + w2 / 2
		right = np.where(r1 > r2, r2, r1)
		result = (right - left)
		return result

	def numpy_iou(self, centroid_gt, centroid_p, dims_gt, dims_p):
		ow = self.numpy_overlap(centroid_p[0], dims_p[0], centroid_gt[0] , dims_gt[0])
		oh = self.numpy_overlap(centroid_p[1], dims_p[1], centroid_gt[1] , dims_gt[1])
		ow = np.where(ow > 0, ow, 0)
		oh = np.where(oh > 0, oh, 0)
		intersection = float(ow) * float(oh)
		area_p = dims_p[0] * dims_p[1]
		area_gt = dims_gt[0] * dims_gt[1]
		union = area_p + area_gt - intersection
		pred_iou = intersection / (float(union) + 0.000001)  # prevent div 0
		return pred_iou

	def read_labels(self):
		depth_label = cv2.imread(self.depth_gt[0], cv2.IMREAD_GRAYSCALE)
		# Read obstacles
		with open(self.obstacles_gt[0],'r') as f:
			obstacles = f.readlines()
		obstacles = [x.strip() for x in obstacles]
		# Label obtacles
		obstacles_label = np.zeros(shape=(5,8,2,7))
		# Anchors
		anchors = np.array([[0.34755122, 0.84069513], 
							[0.14585618, 0.25650666]])
		for obs in obstacles:
			parsed_str_obs = obs.split()
			parsed_obs = np.zeros(shape=(8))
			i = 0
			for n in parsed_str_obs:
				if i < 2:
					parsed_obs[i] = int(n)
				else:
					parsed_obs[i] = float(n)
				i += 1

			# Compute centroid and size bounding box
			w = parsed_obs[4]
			h = parsed_obs[5]
			best_iou = -1.
			best_iou_index = 0
			index = 0
			for anchor in anchors:
				# Compute iou
				pred_iou = self.numpy_iou([0., 0.], [0., 0.], anchor, [w, h])
				if pred_iou > best_iou:
					best_iou = pred_iou
					best_iou_index = index
				index += 1
			# Save labels
			obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), best_iou_index, 0] = 1.0 #confidence
			obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), best_iou_index, 1] = parsed_obs[2] # x
			obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), best_iou_index, 2] = parsed_obs[3] # y
			obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), best_iou_index, 3] = w # w
			obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), best_iou_index, 4] = h # h
			obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), best_iou_index, 5] = parsed_obs[6] # m
			obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), best_iou_index, 6] = parsed_obs[7] # v
		# Yolo out + depth put
		labels = {}
		labels["depth"] = np.expand_dims(depth_label, 2)
		labels["obstacles"] = obstacles_label
		return labels