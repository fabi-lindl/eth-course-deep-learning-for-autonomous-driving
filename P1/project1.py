"""
Script to execute all tasks necessary to solve the problems of project 1. 
"""
import os
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from objects import Visualizer2D
from load_data import load_data


if __name__ == '__main__':
    # Visualizer. 
    vis2d = Visualizer2D()
    
    # Import data. 
    data_path = os.path.join('./data', 'data.p')
    vis2d.data = load_data(data_path)

    # Change color of color_map from BGR to RGB.
    vis2d.bgr_to_rgb_colors()

    # Task 1. Birds eye view of the provided scene. 
    vis2d.birds_eye_view()

    # Task 2. 
    # Create image with semantic point clould labels. 
    vis2d.semantic_labeling(vis2d.data['P_rect_20'], vis2d.data['T_cam0_velo'])
    # Create image with semantic point cloud labels and 3d bboxes. 
    vis2d.semantic_labeling_bboxes(vis2d.data['P_rect_20'], vis2d.data['T_cam0_velo'])

    # Task 3. 
    vis2d.lidar_ray_ids(vis2d.data['P_rect_20'], vis2d.data['T_cam0_velo'])
 