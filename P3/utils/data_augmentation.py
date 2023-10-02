import argparse
import os

import numpy as np

import vispy
import yaml

from dataset import DatasetLoader
from utils.task1 import label2corners
from utils.threedvis import Visualizer
from torchvision import transforms

class LidarTransform(object):
    @staticmethod
    def flip_points(points, axis=0):
        points[:, axis] = -points[:, axis]
        return points

    @staticmethod
    def rotate_points(points, angle):
        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                      [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]])
        return points @ R

    @staticmethod
    def scale_points(points, factor=1.0):
        return points * factor

    def unpack_data(self, data):
        points, boxes = data
        self.box_stack_indices = np.cumsum([len(box) for box in boxes[:-1]])
        return points, np.vstack(boxes)

    def pack_data(self, points, boxes):
        return points, np.split(boxes, self.box_stack_indices, axis=0)

class RandomFlip(LidarTransform):
    def __init__(self, proba=0.5):
        self.proba = proba

    def __call__(self, data):
        if np.random.rand() < self.proba:
            points, boxes = self.unpack_data(data)
            boxes[:, [0, 6]] = -boxes[:, [0, 6]]
            points = self.flip_points(points)
            return self.pack_data(points, boxes)
        else:
            return data

class RandomRotate(LidarTransform):
    def __init__(self, degree_range=10, proba=0.5):
        self.degree_range = degree_range * np.pi / 180
        self.proba = proba

    def __call__(self, data):
        if np.random.rand() < self.proba:
            points, boxes = self.unpack_data(data)
            angle = (np.random.rand() * 2 - 1) * self.degree_range
            boxes[:, :3] = self.rotate_points(boxes[:, :3], angle)
            boxes[:, 6] -= angle
            points = self.rotate_points(points, angle)
            return self.pack_data(points, boxes)
        else:
            return data

class RandomScale(LidarTransform):
    def __init__(self, scale_range=[0.95, 1.05], proba=0.5):
        self.scale_range = scale_range
        self.proba = proba

    def __call__(self, data):
        if np.random.rand() < self.proba:
            points, boxes = self.unpack_data(data)
            factor = np.random.rand() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            boxes[:, :6] = self.scale_points(boxes[:, :6], factor)
            points = self.scale_points(points, factor)
            return self.pack_data(points, boxes)
        else:
            return data

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
    target = data_loader.get_data(0, 'target')

    random_tf = transforms.Compose([
                                    RandomFlip(proba=1),
                                    RandomRotate(degree_range=30, proba=1),
                                    RandomScale(scale_range=[0.5, 1.5], proba=1)
                                    ])
    xyz_tf, [pred_tf, target_tf] = random_tf((np.copy(xyz), [np.copy(pred), np.copy(target)]))

    visualizer = Visualizer(up='y', azimuth=-90)
    visualizer.add_boxes(label2corners(pred), color=[1, 1, 1, 0.2])
    visualizer.add_boxes(label2corners(target))
    visualizer.add_points(xyz.reshape(-1, 3), color=[1, 1, 1, 0.1])
    visualizer.add_boxes(label2corners(pred_tf), color=[1, 0, 0, 0.2])
    visualizer.add_boxes(label2corners(target_tf), color=[1, 1, 0, 1])
    visualizer.add_points(xyz_tf.reshape(-1, 3), color=[1, 0, 0, 0.1])
    vispy.app.run()
