import os, argparse, yaml
import vispy
import numpy as np

from dataset import DatasetLoader
from tests.test import CheckTest
from utils.task1 import label2corners, compute_recall
from utils.task2 import roi_pool
from utils.task3 import sample_proposals
from utils.threedvis import Visualizer

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_path', default='config.yaml')
parser.add_argument('--recordings_dir', default='tests/recordings')
parser.add_argument('--task', type=int)
args = parser.parse_args()

def init(split='val'):
    config = yaml.safe_load(open(args.config_path, 'r'))
    assert (os.path.exists(args.recordings_dir))
    # check = CheckTest(config, args.recordings_dir)
    data_loader = DatasetLoader(config['data'], split)
    return config, data_loader

def load_data(data_loader, scene=0):
    pred = data_loader.get_data(scene, 'detections')
    target = data_loader.get_data(scene, 'target')
    xyz = data_loader.get_data(scene, 'xyz')
    feat = data_loader.get_data(scene, 'features')
    return pred, target, xyz, feat

def calculate_average_recall_task1():
    config, data_loader = init('val')
    recalls = []
    for scene in range(len(data_loader)):
        pred, target, xyz, feat = load_data(data_loader, scene)
        recalls.append(compute_recall(pred, target, 0.5))
        if scene % int(len(data_loader)/100) == 0:
            print('.', end='')
    print('\n', end='')
    print(f'The average recall over the validation set is {np.mean(recalls)}.')

def visualize_task2(data_loader, config, scene=0):
    pred, target, xyz, feat = load_data(data_loader, scene)

    # Computation of task2
    from utils.task2 import roi_pool
    valid_pred, pooled_xyz, pooled_feat = roi_pool(pred=pred,
                                                   xyz=xyz,
                                                   feat=feat,
                                                   config=config['data'])
    large_pred = np.copy(valid_pred)
    # large_pred[:, 6] -= 0.4
    large_pred[:, 3:6] += 2 * config['data']['delta']
    large_pred[:, 1] += config['data']['delta']
    valid_large_corner = label2corners(large_pred)
    valid_corner = label2corners(pred)

    # Visualization
    visualizer = Visualizer(up='y', azimuth=-90)
    visualizer.add_boxes(valid_large_corner, color=[0, 0.5, 1, 1])
    visualizer.add_boxes(valid_corner)
    visualizer.add_points(pooled_xyz.reshape(-1, 3), color=[1, 0, 0, 0.7])
    visualizer.add_points(xyz, color=[1, 1, 1, 0.2])
    vispy.app.run()
    print("Visualization done.")

def visualize_task2_single(data_loader, config, box=0, scene=0):
    pred, target, xyz, feat = load_data(data_loader, scene)

    # Computation of task2
    from utils.task2 import roi_pool
    valid_pred, pooled_xyz, pooled_feat = roi_pool(pred=pred,
                                                   xyz=xyz,
                                                   feat=feat,
                                                   config=config['data'])
    large_pred = np.copy(valid_pred)
    # large_pred[:, 6] -= 0.4
    large_pred[:, 3:6] += 2 * config['data']['delta']
    large_pred[:, 1] += config['data']['delta']
    valid_large_corner = label2corners(large_pred)
    valid_corner = label2corners(pred)

    # Visualization
    visualizer = Visualizer(up='y', azimuth=-90)
    visualizer.add_boxes(valid_large_corner[box], color=[0, 0.5, 1, 1])
    visualizer.add_boxes(valid_corner[box])
    visualizer.add_points(pooled_xyz[box], color=[1, 0, 0, 0.7])
    visualizer.add_points(xyz, color=[1, 1, 1, 0.2])
    vispy.app.run()
    print("Visualization done.")

def visualize_task3(data_loader, config, scene=0):
    pred, target, xyz, feat = load_data(data_loader, scene)
    valid_pred, pooled_xyz, pooled_feat = roi_pool(pred=pred,
                                                   xyz=xyz,
                                                   feat=feat,
                                                   config=config['data'])
    assigned_targets, xyz_prop, feat_prop, ious_prop = sample_proposals(pred, target, pooled_xyz, pooled_feat,
                                                                        config['data'], train=True)

    assigned_targets_corners = label2corners(assigned_targets)
    pred_corners = label2corners(pred)
    target_corners = label2corners(target)
    ass_t_c_fg = assigned_targets_corners[ious_prop >= 0.55]
    ass_t_c_undef = assigned_targets_corners[(ious_prop < 0.55) & (ious_prop >= 0.45)]
    ass_t_c_hbg = assigned_targets_corners[(ious_prop < 0.45) & (ious_prop >= 0.05)]
    ass_t_c_ebg = assigned_targets_corners[ious_prop < 0.05]
    # Visualizaiton
    visualizer = Visualizer(up='y', azimuth=-90)
    visualizer.add_boxes(target_corners, color=[0, 1, 0, 1])
    visualizer.add_boxes(ass_t_c_fg, color=[0, 0.3, 1, 1])
    if ass_t_c_undef.size > 0:
        visualizer.add_boxes(ass_t_c_undef, color=[1, 1, 1, 1])
    visualizer.add_boxes(ass_t_c_hbg, color=[1, 0.1, 0.8, 1]) # pink
    visualizer.add_boxes(ass_t_c_ebg, color=[1, 1, 0, 1])
    visualizer.add_points(xyz_prop.reshape(-1, 3), color=[1, 1, 1, 0.2])
    vispy.app.run()
    print("Visualizaiton done.")

task = 2
scene = 0
split = 'val'
# task 2 settings
visualization_modes = ['single', 'single_all', 'all']
mode = 'single_all'
box = 1

if __name__ == '__main__':
    config, data_loader = init(split)

    if task ==1 :
        calculate_average_recall_task1()

    if task == 2:
        if mode == 'all':
            visualize_task2(data_loader, config, scene)
        elif mode == 'single':
            visualize_task2_single(data_loader, config, box=0, scene=scene)
        elif mode == 'single_all':
            for i in range(100):
                print(f"Box {i}")
                visualize_task2_single(data_loader, config, box=i, scene=scene)

    if task == 3:
        visualize_task3(data_loader, config, scene=scene)
