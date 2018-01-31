import tensorflow as tf
import keras.backend as K
import numpy as np
from DepthMetrics import rmse_metric

bnum = 1
side = 1

def overlap(x1, w1, x2, w2):
    l1 = (x1) - w1 / 2
    l2 = (x2) - w2 / 2
    left = tf.where(K.greater(l1, l2), l1, l2)
    r1 = (x1) + w1 / 2
    r2 = (x2) + w2 / 2
    right = tf.where(K.greater(r1, r2), r2, r1)
    result = right - left
    return result

def numpy_overlap(x1, w1, x2, w2):
    l1 = (x1) - w1 / 2
    l2 = (x2) - w2 / 2
    left = np.where(l1>l2,l1,l2)
    r1 = (x1) + w1 / 2
    r2 = (x2) + w2 / 2
    right = np.where(r1>r2,r2,r1)
    result = (right - left)
    return result

def numpy_iou(top_left_gt, top_left_p, dims_gt, dims_p):

    ow = numpy_overlap(top_left_p[0], int(dims_p[0]) , top_left_gt[0] , dims_gt[0])
    oh = numpy_overlap(top_left_p[1], int(dims_p[1]) , top_left_gt[1] , dims_gt[1])
    ow = np.where(ow > 0, ow, 0)
    oh = np.where(oh > 0, oh, 0)
    intersection = float(ow) * float(oh)
    area_p = int(dims_p[0]) * int(dims_p[1])
    area_gt = dims_gt[0] * dims_gt[1]
    union = area_p + area_gt - intersection
    pred_iou = intersection / (float(union) + 0.000001)  # prevent div 0

    minor_area = np.where(area_p < area_gt, area_p, area_gt)
    is_overlap = np.where( minor_area == intersection, 1,0)

    if pred_iou < 0:
        return 0
    return pred_iou, is_overlap


def recall(y_true, y_pred):
    truth_conf_tensor = K.expand_dims(y_true[:, :, 0], 2)  # tf.slice(y_true, [0, 0, 0], [-1,-1, 0])
    truth_xy_tensor = y_true[:, :, 1:3]  # tf.slice(y_true, [0, 0, 1], [-1,-1, 2])
    truth_wh_tensor = y_true[:, :, 3:5]  # tf.slice(y_true, [0, 0, 3], [-1, -1, 4])

    pred_conf_tensor = K.expand_dims(y_pred[:, :, 0], 2)  # tf.slice(y_pred, [0, 0, 0], [-1, -1, 0])
    # pred_conf_tensor = K.tanh(pred_conf_tensor)
    pred_xy_tensor = y_pred[:, :, 1:3]  # tf.slice(y_pred, [0, 0, 1], [-1, -1, 2])
    pred_wh_tensor = y_pred[:, :, 3:5]  # tf.slice(y_pred, [0, 0, 3], [-1, -1, 4])

    tens = K.greater(truth_conf_tensor, 0.5)

    ave_iou, recall, precision, obj_count, intersection, union, ow, oh, x, y, w, h = iou(truth_xy_tensor[:, :, 0],
                                                                                         truth_xy_tensor[:, :, 1],
                                                                                         truth_wh_tensor[:, :, 0],
                                                                                         truth_wh_tensor[:, :, 1],
                                                                                         pred_xy_tensor[:, :, 0],
                                                                                         pred_xy_tensor[:, :, 1],
                                                                                         pred_wh_tensor[:, :, 0],
                                                                                         pred_wh_tensor[:, :, 1],
                                                                                         tens, pred_conf_tensor)
    return recall

def precision(y_true, y_pred):
    truth_conf_tensor = K.expand_dims(y_true[:, :, 0], 2)  # tf.slice(y_true, [0, 0, 0], [-1,-1, 0])
    truth_xy_tensor = y_true[:, :, 1:3]  # tf.slice(y_true, [0, 0, 1], [-1,-1, 2])
    truth_wh_tensor = y_true[:, :, 3:5]  # tf.slice(y_true, [0, 0, 3], [-1, -1, 4])

    pred_conf_tensor = K.expand_dims(y_pred[:, :, 0], 2)  # tf.slice(y_pred, [0, 0, 0], [-1, -1, 0])
    # pred_conf_tensor = K.tanh(pred_conf_tensor)
    pred_xy_tensor = y_pred[:, :, 1:3]  # tf.slice(y_pred, [0, 0, 1], [-1, -1, 2])
    pred_wh_tensor = y_pred[:, :, 3:5]  # tf.slice(y_pred, [0, 0, 3], [-1, -1, 4])

    tens = K.greater(truth_conf_tensor, 0.5)

    ave_iou, recall, precision, obj_count, intersection, union, ow, oh, x, y, w, h = iou(truth_xy_tensor[:, :, 0],
                                                                                         truth_xy_tensor[:, :, 1],
                                                                                         truth_wh_tensor[:, :, 0],
                                                                                         truth_wh_tensor[:, :, 1],
                                                                                         pred_xy_tensor[:, :, 0],
                                                                                         pred_xy_tensor[:, :, 1],
                                                                                         pred_wh_tensor[:, :, 0],
                                                                                         pred_wh_tensor[:, :, 1],
                                                                                         tens, pred_conf_tensor)
    return precision


def iou_metric(y_true, y_pred):
    truth_conf_tensor = K.expand_dims(y_true[:, :, 0], 2)  # tf.slice(y_true, [0, 0, 0], [-1,-1, 0])
    truth_xy_tensor = y_true[:, :, 1:3]  # tf.slice(y_true, [0, 0, 1], [-1,-1, 2])
    truth_wh_tensor = y_true[:, :, 3:5]  # tf.slice(y_true, [0, 0, 3], [-1, -1, 4])

    pred_conf_tensor = K.expand_dims(y_pred[:, :, 0], 2)  # tf.slice(y_pred, [0, 0, 0], [-1, -1, 0])
    # pred_conf_tensor = K.tanh(pred_conf_tensor)
    pred_xy_tensor = y_pred[:, :, 1:3]  # tf.slice(y_pred, [0, 0, 1], [-1, -1, 2])
    pred_wh_tensor = y_pred[:, :, 3:5]  # tf.slice(y_pred, [0, 0, 3], [-1, -1, 4])

    tens = K.greater(truth_conf_tensor, 0.5)

    ave_iou, recall, precision, obj_count, intersection, union, ow, oh, x, y, w, h = iou(truth_xy_tensor[:, :, 0],
                                                                                         truth_xy_tensor[:, :, 1],
                                                                                         truth_wh_tensor[:, :, 0],
                                                                                         truth_wh_tensor[:, :, 1],
                                                                                         pred_xy_tensor[:, :, 0],
                                                                                         pred_xy_tensor[:, :, 1],
                                                                                         pred_wh_tensor[:, :, 0],
                                                                                         pred_wh_tensor[:, :, 1],
                                                                                         tens, pred_conf_tensor)
    return ave_iou

def mean_metric(y_true, y_pred):
    truth_m_tensor = K.expand_dims(y_true[:, :, 5], 2)
    pred_m_tensor = K.expand_dims(y_pred[:, :, 5], 2)

    return rmse_metric(truth_m_tensor,pred_m_tensor)

def variance_metric(y_true, y_pred):
    truth_v_tensor = K.expand_dims(y_true[:, :, 6], 2)
    pred_v_tensor = K.expand_dims(y_pred[:, :, 6], 2)

    return rmse_metric(truth_v_tensor, pred_v_tensor)


def iou(x_true, y_true, w_true, h_true, x_pred, y_pred, w_pred, h_pred, t, pred_confid_tf):
    x_true = K.expand_dims(x_true, 2)
    y_true = K.expand_dims(y_true, 2)
    w_true = K.expand_dims(w_true, 2)
    h_true = K.expand_dims(h_true, 2)
    x_pred = K.expand_dims(x_pred, 2)
    y_pred = K.expand_dims(y_pred, 2)
    w_pred = K.expand_dims(w_pred, 2)
    h_pred = K.expand_dims(h_pred, 2)

    xoffset = K.expand_dims(tf.convert_to_tensor(np.asarray([0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7], dtype=np.float32)),1)
    yoffset = K.expand_dims(tf.convert_to_tensor(np.asarray([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4], dtype=np.float32)),1)


    # xoffset = K.cast_to_floatx((np.tile(np.arange(side),side)))
    # yoffset = K.cast_to_floatx((np.repeat(np.arange(side),side)))
    x = tf.where(t, x_pred, K.zeros_like(x_pred))
    y = tf.where(t, y_pred, K.zeros_like(y_pred))
    w = tf.where(t, w_pred, K.zeros_like(w_pred))
    h = tf.where(t, h_pred, K.zeros_like(h_pred))

    ow = overlap(x + xoffset, w * 256. , x_true + xoffset, w_true * 256.)
    oh = overlap(y + yoffset, h * 160., y_true + yoffset, h_true * 256.)

    ow = tf.where(K.greater(ow, 0), ow, K.zeros_like(ow))
    oh = tf.where(K.greater(oh, 0), oh, K.zeros_like(oh))
    intersection = ow * oh
    union = w * 256. * h * 160. + w_true * 256. * h_true * 160.  - intersection + K.epsilon()  # prevent div 0

    #
    # find best iou among bboxs
    # iouall shape=(-1, bnum*gridcells)
    iouall = intersection / union
    obj_count = K.sum(tf.where(t, K.ones_like(x_true), K.zeros_like(x_true)))

    ave_iou = K.sum(iouall) / (obj_count + 0.0000001)
    recall_t = K.greater(iouall, 0.5)
    # recall_count = K.sum(tf.select(recall_t, K.ones_like(iouall), K.zeros_like(iouall)))

    fid_t = K.greater(pred_confid_tf, 0.3)
    recall_count_all = K.sum(tf.where(fid_t, K.ones_like(iouall), K.zeros_like(iouall)))

    #  
    obj_fid_t = tf.logical_and(fid_t, t)
    obj_fid_t = tf.logical_and(fid_t, recall_t)
    effevtive_iou_count = K.sum(tf.where(obj_fid_t, K.ones_like(iouall), K.zeros_like(iouall)))

    recall = effevtive_iou_count / (obj_count + 0.00000001)
    precision = effevtive_iou_count / (recall_count_all + 0.0000001)
    return ave_iou, recall, precision, obj_count, intersection, union, ow, oh, x, y, w, h


# return obj_count, ave_iou, bestiou




def yolo_conf_loss(y_true, y_pred, t):
    real_y_true = tf.where(t, y_true, K.zeros_like(y_true))
    pobj = K.sigmoid(y_pred)
    lo = K.square(real_y_true - pobj)
    value_if_true = 5.0 * (lo)
    value_if_false = 0.05 * (lo)
    loss1 = tf.where(t, value_if_true, value_if_false)

    loss = K.mean(loss1)
    return loss

def yoloxyloss(y_true, y_pred, t):
    #real_y_true = tf.where(t, y_true, K.zeros_like(y_true))
    lo = K.square(y_true - y_pred) + 0.05 * K.square(0.5 -y_pred)
    value_if_true = lo
    value_if_false = K.zeros_like(y_true)
    loss1 = tf.where(t, value_if_true, value_if_false)
    objsum = K.sum(y_true)
    return K.sum(loss1)/(objsum+0.0000001)

def yolo_wh_loss(y_true,y_pred,t):
    #real_y_true = tf.where(t, y_true, K.zeros_like(y_true))
    lo = K.square(y_true - y_pred) + 0.05* K.square(0.5 - y_pred)
    #lo = K.square(y_true - y_pred) + 0.3 * K.square(0.5 - y_pred)
    value_if_true = lo
    value_if_false = K.zeros_like(y_true)
    loss1 = tf.where(t, value_if_true, value_if_false)
    objsum = K.sum(y_true)
    return K.sum(loss1) / (objsum + 0.0000001)

def yolo_regressor_loss(y_true,y_pred,t):
    #real_y_true = tf.where(t, y_true, K.zeros_like(y_true))
    lo = K.square(y_true - y_pred) #+ 0.15 * K.square(0.5 - y_pred)
    # lo = K.square(y_true - y_pred) + 0.3 * K.square(0.5 - y_pred)
    value_if_true = lo
    value_if_false = K.zeros_like(y_true)
    loss1 = tf.where(t, value_if_true, value_if_false)

    objsum = K.sum(y_true)
    return K.sum(loss1) / (objsum + 0.0000001)

def yolo_v1_loss(y_true, y_pred):
    # Y_PRED is Batchx40x7 tensor. y_true is a 40x7 tensor

    truth_conf_tensor = K.expand_dims(y_true[:,:,0],2)#tf.slice(y_true, [0, 0, 0], [-1,-1, 0])
    truth_xy_tensor = y_true[:,:,1:3]#tf.slice(y_true, [0, 0, 1], [-1,-1, 2])
    truth_wh_tensor = y_true[:,:,3:5]#tf.slice(y_true, [0, 0, 3], [-1, -1, 4])
    truth_m_tensor = K.expand_dims(y_true[:,:,5],2)#tf.slice(y_true, [0, 0, 5], [-1, -1, 5])
    truth_v_tensor = K.expand_dims(y_true[:,:,6],2)#tf.slice(y_true, [0, 0, 6], [-1, -1, 6])

    pred_conf_tensor = K.expand_dims(y_pred[:,:,0],2)#tf.slice(y_pred, [0, 0, 0], [-1, -1, 0])
    #pred_conf_tensor = K.tanh(pred_conf_tensor)
    pred_xy_tensor = y_pred[:,:,1:3]#tf.slice(y_pred, [0, 0, 1], [-1, -1, 2])
    pred_wh_tensor = y_pred[:,:,3:5]#tf.slice(y_pred, [0, 0, 3], [-1, -1, 4])
    pred_m_tensor = K.expand_dims(y_pred[:,:,5],2)#tf.slice(y_pred, [0, 0, 5], [-1, -1, 5])
    pred_v_tensor = K.expand_dims(y_pred[:,:,6],2)#tf.slice(y_pred, [0, 0, 6], [-1, -1, 6])

    truth_xy_tensor = tf.Print(truth_xy_tensor, [truth_xy_tensor[:, 14:20, 0]], message='truth_xy', summarize=30)
    pred_xy_tensor = tf.Print(pred_xy_tensor, [pred_xy_tensor[:, 14:20, 0]], message='pred_xy', summarize=30)

    tens = K.greater(K.sigmoid(truth_conf_tensor), 0.5)
    tens_2d = K.concatenate([tens,tens], axis=-1)

    conf_loss = yolo_conf_loss(truth_conf_tensor, pred_conf_tensor,tens)
    xy_loss = yoloxyloss(truth_xy_tensor,pred_xy_tensor,tens_2d)
    wh_loss = yolo_wh_loss(truth_wh_tensor,pred_wh_tensor,tens_2d)
    m_loss = yolo_regressor_loss(truth_m_tensor,pred_m_tensor,tens)
    v_loss = yolo_regressor_loss(truth_v_tensor,pred_v_tensor,tens)

    loss = 2.0 * conf_loss + 0.25 * xy_loss + 0.25 * wh_loss + 1.5 * m_loss + 1.25 * v_loss # loss v1
    #loss = 2.0 * conf_loss + 0.1 * xy_loss + 1.0 * wh_loss + 5.0 * m_loss + 2.5 * v_loss  # loss v2


    return loss