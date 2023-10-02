import numpy as np

from utils.task1 import get_iou

def nms(pred, score, threshold):
    """
    Task 5
    Implement NMS to reduce the number of predictions per frame with a threshold
    of 0.1. The IoU should be calculated only on the BEV.
    input
        pred (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
        score (N,) confidence scores
        threshold (float) upper bound threshold for NMS
    output
        s_f (M,7) 3D bounding boxes after NMS
        c_f (M,1) corresopnding confidence scores
    """
    pred_2d = np.copy(pred)  # array with equal y and h to emulate 2d BEV IoU
    pred_2d[:, 1] = 0  # set y to 0
    pred_2d[:, 3] = 1  # set h to 1
    pred_mask = np.ones(score.shape, dtype=bool)  # mask to keep track which preds are excluded
    final_inds = []  # list of final indices. Not a mask to get a sorted output
    while pred[pred_mask].size:
        max_score_ind = np.where(score == np.max(score[pred_mask]))[0][0]  # np.where return tuple with first element being an aray
        final_inds.append(max_score_ind)  # keep track of best candidates
        pred_mask[max_score_ind] = False  # exclude current candidate
        ious = get_iou(pred_2d[pred_mask], pred_2d[max_score_ind].reshape(1, -1)).flatten()  # calculate ious
        pred_mask[pred_mask] = ~(ious >= threshold)  # exclude overlapping predictions

    return pred[final_inds], score[final_inds].reshape(-1, 1)
