import cv2
import numpy as np
import os
from glob import glob
import math

class Obstacle(object):
    def __init__(self, x, y, w, h, segmentation, depth_gt):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.segmentation = segmentation
        self.valid_points = -1
        self.depth_mean, self.depth_variance = self.compute_depth_stats(depth_gt)

    def compute_depth_stats(self, depth):

        roi_depth = depth[self.y:self.y+self.h, self.x:self.x+self.w]
        roi_segm = self.segmentation[self.y:self.y+self.h, self.x:self.x+self.w]

        #depth = depth.astype("float32")
        #depth_c = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)

        mean_depth = 0
        squared_sum = 0
        valid_points = 0

        for y in range(0,self.h):
            for x in range(0,self.w):
                if roi_segm[y,x] > 0:
                    mean_depth += roi_depth.item(y,x)
                    squared_sum += roi_depth.item(y,x)**2.
                    #cv2.circle(depth_c, (self.x+x,self.y+y), 2, (0,255,0))
                    valid_points += 1
                    #cv2.circle(depth_c, (self.x+x, self.y+y), 2, (255, 0, 0))
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

        mean, var = self.compute_depth_stats(estimated_depth)

        mean_rmse = (self.depth_mean - mean)**2
        mean_variance = (self.depth_variance - var)**2

        #print ("GT depth: %.02f, Predicted depth: %.02f"%(self.depth_mean, mean))

        return math.sqrt(mean_rmse), math.sqrt(mean_variance), self.valid_points

def depth_to_meters_airsim(depth):

    depth = depth.astype(np.float64)

    for i in range(0,depth.shape[0]):
        for j in range(0, depth.shape[1]):
            depth[i,j] = (-4.586e-09 * (depth[i,j] ** 4.)) + (3.382e-06 * (depth[i,j] ** 3.)) - (0.000105 * (depth[i,j] ** 2.)) + (0.04239 * depth[i,j]) + 0.04072

    return depth

def find_obstacles(depth, segmentation, depth_thr, segm_thr, f_segm_thr, IMG_WIDTH = 256., IMG_HEIGHT = 160., X_SECTORS = 8., Y_SECTORS = 5., MIN_OBS_AREA = 30, test_obstacles = False):
    # convert depth in meters
    depth = depth_to_meters_airsim(depth)
    # threshold depth
    retval_d, depth_mask = cv2.threshold(depth, depth_thr, 1., cv2.THRESH_BINARY_INV)
    # threshold segmentation
    retval, obstacles_mask = cv2.threshold(segmentation, int(segm_thr), 1., f_segm_thr)

    # remove contours for thresholded segmentation
    obstacles_mask = obstacles_mask.astype(np.float32)*depth_mask

    im2, contours, hierarchy = cv2.findContours(obstacles_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
    contours_img = np.zeros(shape=obstacles_mask.shape)
    contours_img = cv2.drawContours(contours_img, contours, -1, (1, 1, 1), 2)

    obstacles_mask = (obstacles_mask) - contours_img.astype(np.uint8)

    obstacles_annotations = []
    i = 0
    time_to_show = 2 #used on test

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # hull[i] = cv2.convexHull(cnt)
        #cv2.imshow("obstacles mask", depth_mask * depth * obstacles_mask / depth_thr)
        obstacle = Obstacle(x, y, w, h, obstacles_mask, depth_mask * depth * (obstacles_mask))

        #x,y are top-left rectangle vertex coordinats. I need the center coordinates
        x_c = x + w/2.
        y_c = y + h/2.

        idx_x = int(math.floor(x_c / (IMG_WIDTH / X_SECTORS)))
        idx_y = int(math.floor(y_c / (IMG_HEIGHT / Y_SECTORS)))

        #Normalize x_c, y_c between 0,1 parameterized to idx_x and idx_y

        x_c = x_c / (IMG_WIDTH / X_SECTORS) - idx_x
        y_c = y_c / (IMG_HEIGHT / Y_SECTORS) - idx_y

        #Normalize w,h to [0..1]
        w_o = w / IMG_WIDTH
        h_o = h / IMG_HEIGHT

        mean = obstacle.depth_mean / 39.75
        variance = obstacle.depth_variance * 0.01 / 39.75

        if obstacle.valid_points > MIN_OBS_AREA and mean > 0:
            cv2.rectangle(segmentation, (x, y), (x + w, y + h), (0, 255, 0), 2)
            error_text = ("%.2f" % (mean))
            cv2.putText(segmentation, error_text, (x + w / 2, y + h / 2), cv2.FONT_HERSHEY_PLAIN, 1, 255)
            obstacles_annotations.append(str(idx_x) +' '+ str(idx_y)+' '+'{:.3f}'.format(x_c)+' '+'{:.3f}'.format(y_c)+' '+'{:.3f}'.format(w_o)+' '+'{:.3f}'.format(h_o)+' '+'{:.3f}'.format(mean)+' '+'{:.3f}'.format(variance)+'\n')
            time_to_show = 50
        i += 1
    if test_obstacles:
        #cv2.imshow("test", segmentation)
        print obstacles_annotations
        cv2.waitKey(time_to_show)
        return None
    else:
        #cv2.imshow("test", segmentation)
        #cv2.waitKey(2)
        return obstacles_annotations

dataset_dir = "/home/isarlab/Datasets/UnrealDataset/"

seq_dirs = ['00_D','01_D','02_D','03_D','04_D','05_D','06_D','07_D','08_D','09_D','10_D','11_D','13_D','14_D','15_D','16_D','17_D','18_D','19_D','20_D']
#seq_dirs = ['11_D','13_D','14_D','15_D','16_D','17_D','18_D','19_D','20_D']
seq_cnt = 0
for seq in seq_dirs:
    seq_cnt += 1
    depth_gt_paths = sorted(glob(os.path.join(dataset_dir, seq, 'depth', '*' + '.png')))
    seg_paths = sorted(glob(os.path.join(dataset_dir, seq, 'segmentation', '*' + '.png')))

    target_dir = os.path.join(dataset_dir, seq, 'obstacles_20m')
    os.mkdir(target_dir)

    for gt_path, seg_path in zip(depth_gt_paths, seg_paths):
        gt = cv2.imread(gt_path, 0)
        seg = cv2.imread(seg_path, 0)

        segm_thr = 55
        depth_thr = 20
        thr_strategy = cv2.THRESH_BINARY

        if seq_cnt < 13:
            segm_thr = 15
            thr_strategy = cv2.THRESH_BINARY_INV

        obstacles_annotation = find_obstacles(gt,seg, depth_thr, segm_thr,thr_strategy, test_obstacles= False)


        if obstacles_annotation is not None:
            print gt_path
            file_name_tmp = os.path.split(gt_path)[1]
            file_name = os.path.splitext(file_name_tmp)[0]+'.txt'
            obs_file = open(os.path.join(target_dir, file_name),'w')

            for annotation in obstacles_annotation:
                obs_file.write(annotation)
            obs_file.close()
