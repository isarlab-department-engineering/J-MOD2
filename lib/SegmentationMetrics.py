import numpy as np

class SegmentationMetrics:
    def __init__(self):
        self.pixel_accuracy_res = 0
        self.mean_accuracy_res = 0
        self.mean_IU_res = 0
        self.freq_weight_IU_res =0
        self.n_eval = 0

    def update(self, pred, gt):
        self.n_eval += 1
        self.pixel_accuracy_res += self.pixel_accuracy(pred,gt)
        self.mean_IU_res += self.mean_IU(pred,gt)
        self.freq_weight_IU_res += self.frequency_weighted_IU(pred,gt)
        self.mean_accuracy_res += self.mean_accuracy(pred,gt)
    def get_stats(self):
        div = self.n_eval
        return self.pixel_accuracy_res/div, self.mean_accuracy_res/div, self.mean_IU_res/div, self.freq_weight_IU_res/div
    def pixel_accuracy(self,eval_segm, gt_segm):
        '''
        sum_i(n_ii) / sum_i(t_i)
        '''

        self.check_size(eval_segm, gt_segm)

        cl, n_cl = self.extract_classes(gt_segm)
        eval_mask, gt_mask = self.extract_both_masks(eval_segm, gt_segm, cl, n_cl)

        sum_n_ii = 0
        sum_t_i = 0

        for i, c in enumerate(cl):
            curr_eval_mask = eval_mask[i, :, :]
            curr_gt_mask = gt_mask[i, :, :]

            sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            sum_t_i += np.sum(curr_gt_mask)

        if (sum_t_i == 0):
            pixel_accuracy_ = 0
        else:
            pixel_accuracy_ = sum_n_ii / sum_t_i

        return pixel_accuracy_

    def mean_accuracy(self, eval_segm, gt_segm):
        '''
        (1/n_cl) sum_i(n_ii/t_i)
        '''

        self.check_size(eval_segm, gt_segm)

        cl, n_cl = self.extract_classes(gt_segm)
        eval_mask, gt_mask = self.extract_both_masks(eval_segm, gt_segm, cl, n_cl)

        accuracy = list([0]) * n_cl

        for i, c in enumerate(cl):
            curr_eval_mask = eval_mask[i, :, :]
            curr_gt_mask = gt_mask[i, :, :]

            n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            t_i = np.sum(curr_gt_mask)

            if (t_i != 0):
                accuracy[i] = n_ii / t_i

        mean_accuracy_ = np.mean(accuracy)
        return mean_accuracy_

    def mean_IU(self,eval_segm, gt_segm):
        '''
        (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
        '''

        self.check_size(eval_segm, gt_segm)

        cl, n_cl = self.union_classes(eval_segm, gt_segm)
        _, n_cl_gt = self.extract_classes(gt_segm)
        eval_mask, gt_mask = self.extract_both_masks(eval_segm, gt_segm, cl, n_cl)

        IU = list([0]) * n_cl

        for i, c in enumerate(cl):
            curr_eval_mask = eval_mask[i, :, :]
            curr_gt_mask = gt_mask[i, :, :]

            if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
                continue

            n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            t_i = np.sum(curr_gt_mask)
            n_ij = np.sum(curr_eval_mask)

            IU[i] = n_ii / (t_i + n_ij - n_ii)

        mean_IU_ = np.sum(IU) / n_cl_gt
        return mean_IU_

    def frequency_weighted_IU(self, eval_segm, gt_segm):
        '''
        sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
        '''

        self.check_size(eval_segm, gt_segm)

        cl, n_cl = self.union_classes(eval_segm, gt_segm)
        eval_mask, gt_mask = self.extract_both_masks(eval_segm, gt_segm, cl, n_cl)

        frequency_weighted_IU_ = list([0]) * n_cl

        for i, c in enumerate(cl):
            curr_eval_mask = eval_mask[i, :, :]
            curr_gt_mask = gt_mask[i, :, :]

            if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
                continue

            n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            t_i = np.sum(curr_gt_mask)
            n_ij = np.sum(curr_eval_mask)

            frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

        sum_k_t_k = self.get_pixel_area(eval_segm)

        frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
        return frequency_weighted_IU_

    '''
    Auxiliary functions used during evaluation.
    '''

    def get_pixel_area(self,segm):
        return segm.shape[0] * segm.shape[1]

    def extract_both_masks(self,eval_segm, gt_segm, cl, n_cl):
        eval_mask = self.extract_masks(eval_segm, cl, n_cl)
        gt_mask = self.extract_masks(gt_segm, cl, n_cl)

        return eval_mask, gt_mask

    def extract_classes(self,segm):
        cl = np.unique(segm)

        if cl[-1] == 255:
             cl = cl[0:-1] #remove 255

        n_cl = len(cl)

        return cl, n_cl

    def union_classes(self,eval_segm, gt_segm):
        eval_cl, _ = self.extract_classes(eval_segm)
        gt_cl, _ = self.extract_classes(gt_segm)

        cl = np.union1d(eval_cl, gt_cl)
        n_cl = len(cl)

        return cl, n_cl

    def extract_masks(self,segm, cl, n_cl):
        h, w = self.segm_size(segm)
        masks = np.zeros((n_cl, h, w))

        for i, c in enumerate(cl):
            masks[i, :, :] = segm == c

        return masks

    def segm_size(self,segm):
        try:
            height = segm.shape[0]
            width = segm.shape[1]
        except IndexError:
            raise

        return height, width

    def check_size(self,eval_segm, gt_segm):
        h_e, w_e = self.segm_size(eval_segm)
        h_g, w_g = self.segm_size(gt_segm)

        if (h_e != h_g) or (w_e != w_g):
            raise RuntimeError

    class EvalSegErr(Exception):
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return repr(self.value)