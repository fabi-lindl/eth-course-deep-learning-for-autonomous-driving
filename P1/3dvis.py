import numpy as np
import os
import vispy
from vispy.scene import visuals, SceneCanvas
from load_data import load_data
from objects import Visualizer2D

class Visualizer():
    def __init__(self):
        self.canvas = SceneCanvas(keys='interactive', show=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.view = vispy.scene.widgets.ViewBox(border_color='white',
                        parent=self.canvas.scene)
        self.grid.add_widget(self.view, 0, 0)

        # Point Cloud Visualizer
        self.sem_vis = visuals.Markers()
        self.view.camera = vispy.scene.cameras.TurntableCamera(up='z', azimuth=90)
        self.view.add(self.sem_vis)
        visuals.XYZAxis(parent=self.view.scene)
        
        # Object Detection Visualizer
        self.obj_vis = visuals.Line()
        self.view.add(self.obj_vis)
        self.connect = np.asarray([[0,1],[0,3],[0,4],
                                   [2,1],[2,3],[2,6],
                                   [5,1],[5,4],[5,6],
                                   [7,3],[7,4],[7,6]])

    def update(self, points, colors):
        '''
        Prameters:
        points: Point cloud data, numpy array [[a, b, c] [d, e, f] ... []]
        colors: Point colors, list of 3N lists [[a, b, c], ... ]. 

        Task 2: Change this function such that each point
        is colored depending on its semantic label
        '''
        self.sem_vis.set_data(points, size=3, face_color=colors)
    
    def update_boxes(self, corners):
        '''
        :param corners: corners of the bounding boxes
                        shape (N, 8, 3) for N boxes
        (8, 3) array of vertices for the 3D box in
        following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        If you plan to use a different order, you can
        change self.connect accordingly.
        '''
        for i in range(corners.shape[0]):
            connect = np.concatenate((connect, self.connect+8*i), axis=0) \
                      if i>0 else self.connect
        
        self.obj_vis.set_data(corners.reshape(-1,3), connect=connect, width=2, color=[0,1,0,1])

if __name__ == '__main__':
    # Load the data.
    data = load_data('data/data.p') # Change to data.p for your final submission
    
    # Create vis 2d object.
    vis2d = Visualizer2D()
    vis2d.data = data

    # Color map.
    # Change color of color_map from BGR to RGB and scale values for the interval [0, 1].
    colorMap = vis2d.bgr_to_rgb_colors(scale=1)
    colors = []
    for i in data['sem_label']:
        colors.append(colorMap[i[0]])

    # Compute 3d bboxes for cars.
    # Compute inverse matrix for cs transformation.
    tm = np.linalg.inv(data['T_cam0_velo'])
    objBboxs = vis2d.bbox_coordinates('lidar', tm)

    # Visualize results.
    visualizer = Visualizer()
    visualizer.update(data['velodyne'][:,:3], colors)
    
    '''
    Task 2: Compute all bounding box corners from given
    annotations. You can visualize the bounding boxes using
    visualizer.update_boxes(corners)
    '''
    visualizer.update_boxes(objBboxs)

    # Run visualizer.
    vispy.app.run()
