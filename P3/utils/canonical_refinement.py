import argparse
import os

import numpy as np
import vispy
import yaml

from dataset import DatasetLoader
from utils.task1 import label2corners
from utils.task2 import roi_pool
from utils.threedvis import Visualizer

def transform_to_origin(points, boxes):
    '''
    Transforms pooled points to origin

    INPUT
        points: 512 points per box (N, 512, 3) N is the number of boxes
        boxes: parameters of a bounding box (N, 7)

    RETURNS
        points_trans_rot (N, 512, 3) transformed to the orignin
    '''
    coords = boxes[:, :3]
    r =  boxes[:, 6] + np.pi/2
    # rotation matrices
    R = np.transpose(np.array([[np.cos(r), np.zeros(r.shape), np.sin(r)],
                               [np.zeros(r.shape), np.ones(r.shape), np.zeros(r.shape)],
                               [-np.sin(r), np.zeros(r.shape), np.cos(r)]]), (2, 0, 1))
    points_trans = points - coords[:, np.newaxis, :]
    points_trans_rot = points_trans @ R
    return points_trans_rot

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_path', default='config.yaml')
    parser.add_argument('--recordings_dir', default='tests/recordings')
    parser.add_argument('--task', type=int)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_path, 'r'))
    assert (os.path.exists(args.recordings_dir))
    # check = CheckTest(config, args.recordings_dir)
    data_loader = DatasetLoader(config['data'], 'val')
    pred = data_loader.get_data(0, 'detections')
    xyz = data_loader.get_data(0, 'xyz')
    feat = data_loader.get_data(0, 'features')

    valid_pred, pooled_xyz, pooled_feat = roi_pool(pred=pred,
                                                   xyz=xyz,
                                                   feat=feat,
                                                   config=config['data'])

    points_canonical = transform_to_origin(pooled_xyz, valid_pred)

    cpred = label2corners(pred)

    box = 13
    visualizer = Visualizer(up='y', azimuth=-90)
    visualizer.add_points(points_canonical[box].reshape(-1, 3))
    visualizer.add_points(pooled_xyz[box].reshape(-1, 3))
    visualizer.add_boxes(cpred[box], color=[1, 0, 0, 1])
    vispy.app.run()
