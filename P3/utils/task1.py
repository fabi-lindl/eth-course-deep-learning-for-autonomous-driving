import itertools

import numpy as np
from numpy import cos, sin, pi
# from utils.threedvis import Visualizer
# import matplotlib.pyplot as plt
# import vispy

import shapely.geometry
import shapely.affinity

DEBUG = True

def calc_iou(rect1_coords, rect2_coords):
    '''
    Calculate iou between two boxes. Since the boxes are only rotated around the y-axis,
    first calculate iou of the ground plane, and then multiply by height to get volume.
    input
        rect1_coords: box given by 8 corners (8, 3)
        rect2_coords: box given by 8 corners (8, 3)
    output
        iou between the two boxes
    '''
    y1_low = rect1_coords[4, 1]  # index 4 is the first corner in the bottom rectangle, 1=y
    y1_high = rect1_coords[0, 1]  # index 0 is the first corner in the top rectangle
    y2_low = rect2_coords[4, 1]
    y2_high = rect2_coords[0, 1]
    y1_height = y1_high - y1_low
    y2_height = y2_high - y2_low

    if y1_low < y2_high and y2_low < y1_high:
        y_overlap = min(y1_high, y2_high) - max(y1_low, y2_low)
    else:
        # print("No overlap!")
        return 0.0

    rect1_coords2d = rect1_coords[4:, :][:, [0, 2]]  # only take the bottom 4 coordinates and only x and z
    rect2_coords2d = rect2_coords[4:, :][:, [0, 2]]  # only take the bottom 4 coordinates and only x and z
    rect1 = shapely.geometry.polygon.Polygon(rect1_coords2d)
    rect2 = shapely.geometry.polygon.Polygon(rect2_coords2d)
    intersec_area = rect1.intersection(rect2).area
    intersec_vol = intersec_area*y_overlap
    return intersec_vol / (rect1.area*y1_height + rect2.area*y2_height - intersec_vol)

def label2corners(label):
    '''
    Task 1
    input
        label (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
    output
        corners (N,8,3) corner coordinates in the rectified reference frame
    '''
    x,y,z,h,w,l,r = np.hsplit(label, label.shape[1])
    r = -r + np.pi/2
    dx = w/2
    dz = l/2
    dxcos = dx*cos(r)
    dxsin = dx*sin(r)
    dzcos = dz*cos(r)
    dzsin = dz*sin(r)

    '''
    calculate corners
    following order:
        1 -------- 0
       /|         /|
      2 -------- 3 .
      | |        | |
      . 5 -------- 4
      |/         |/
      6 -------- 7
    '''
    corners = np.zeros((label.shape[0], 8, 3))
    # top corners
    corners[:, 0, :] = np.hstack([x + dxcos - dzsin, y, z + dzcos + dxsin])
    corners[:, 1, :] = np.hstack([x - dxcos - dzsin, y, z + dzcos - dxsin])
    corners[:, 2, :] = np.hstack([x - dxcos + dzsin, y, z - dzcos - dxsin])
    corners[:, 3, :] = np.hstack([x + dxcos + dzsin, y, z - dzcos + dxsin])
    # bottom corners
    corners[:, 4, :] = np.hstack([x + dxcos - dzsin, y - h, z + dzcos + dxsin])
    corners[:, 5, :] = np.hstack([x - dxcos - dzsin, y - h, z + dzcos - dxsin])
    corners[:, 6, :] = np.hstack([x - dxcos + dzsin, y - h, z - dzcos - dxsin])
    corners[:, 7, :] = np.hstack([x + dxcos + dzsin, y - h, z - dzcos + dxsin])

    return corners

def label2corners_shapely(label):
    '''
    Same as label2corners, but uses shapely
    RETURNS DIFFERENT ORDER OF POINTS
    '''
    corners = np.zeros((label.shape[0], 8, 3))
    for i, sample in enumerate(label):
        x, y, z, h, w, l, r = sample
        box = shapely.geometry.box(-w/2.0, -l/2.0, w/2.0, l/2.0)
        rotated_box = shapely.affinity.rotate(box, r + np.pi/2, use_radians=True)
        xy = shapely.affinity.translate(rotated_box, x, z).exterior.xy
        corners[i, :4, 0] = xy[0][0:4]
        corners[i, 4:, 0] = xy[0][0:4]
        corners[i, :4, 2] = xy[1][0:4]
        corners[i, 4:, 2] = xy[1][0:4]

    y_all = label[:, 1].reshape(-1, 1)
    h_all = label[:, 3].reshape(-1, 1)
    corners[:, :4, 1] = np.repeat(y_all, 4, axis=1)
    corners[:, 4:, 1] = np.repeat(y_all - h_all, 4, axis=1)

    return corners


def bbox_coordinates(objects, tm=np.eye(4)):
    """
    *** function from project 1. Used for debugging purposes ***
    Parameters:
    tm: Cs transformation matrix as 2d numpy array.
    -----------
    Computes the bbox coordinates either for a 2d image plane (cams) or 3d lidar cs.
    Returns:
    -------
    Lidar: Numpy array [objNumber x 8 x 3]. 8 edges and 3 as x, y, z coordinates.
    Cam  : Numpy array [objNumber x 8 x 3]. 8 edges and 3 as x, y, z coordinates.
           x distance from the left image edge. y distance from the bottom.
           z hom. coordinate (ignore this one).
    """
    # Compute 3d bboxes for cars.
    numObjs = len(objects)
    # 8 edge points with 3 coordinates (x, y, z) each.
    # Create [numObjsx8x3] array to store the bbox edge points.
    objbboxes = np.zeros(numObjs * 8 * 3).reshape(numObjs, 8, 3)
    idx = 0
    # Loop over all objects and compute the bbox coordinate points
    # projected onto the camera image plane.
    for obj in objects:
        # Create storage array [4x8] for edge point coordinates.
        bboxEdges = np.zeros(32).reshape(4, 8)
        # Extract data from object data format.
        h = obj[3]  # height
        w = obj[4]  # width
        l = obj[5]  # length
        # Add 90 degrees to the given rotation to transform the
        # roation into the 2d coordinate system used for computing
        # the edge positions in the x, z plane.
        yrot = np.pi / 2 + obj[6]  # Rotation around y axis.
        # Bbox edges.
        xd = w / 2  # x-distance (lidar cs)
        zd = l / 2  # z-distance (lidar cs)
        # Set up rotation matrix in the 2d camera x-, z-plane.
        rot2d = np.array([
            [np.cos(yrot), -np.sin(yrot)],
            [np.sin(yrot), np.cos(yrot)]]
        )
        cnt = 0
        for i in range(1, -2, -2):  # x coordinate
            for j in range(1, -2, -2):  # z coordinate
                if i < 0:
                    # Required for the edge point ordering. Otherwise bboxes are drawn incorrectly.
                    # Necessary for the pillow polygon function.
                    j *= -1
                # Point coordinates in the x-, z-plane.
                ov = np.array([xd * i, zd * j])
                ov = np.matmul(np.transpose(rot2d), ov)  # Rotation.
                x = obj[0]+ ov[0]  # Object center plus offset.
                z = obj[2] + ov[1]  # Object center plus offset.
                # Store computed edge point coordinates.
                bboxEdges[:, cnt] = np.array([x, obj[1], z, 1])  # bottom point
                bboxEdges[:, cnt + 4] = np.array([x, obj[1] - h, z, 1])  # top point
                cnt += 1


        bblidar = np.matmul(tm, bboxEdges)
        bblidar = np.delete(bblidar, 3, 0)
        bblidar = np.transpose(bblidar)
        objbboxes[idx] = bblidar
        idx += 1

    return objbboxes

# def visualize_corners(corners, color=[0, 1, 0, 1]):
#     visualizer = Visualizer()
#     visualizer.add_boxes(corners, color=color)
#     vispy.app.run()

def get_iou(pred, target):
    '''
    Task 1
    input
        pred (N,7) 3D bounding box corners
        target (N,7) 3D bounding box corners
    output
        iou (N,M) pairwise 3D intersection-over-union
    '''
    corners_pred = label2corners(pred)
    corners_target = label2corners(target)

    # if False:
    #     visualizer = Visualizer()
    #     visualizer.add_boxes(corners_pred.reshape(-1, 3), color=[1, 0, 0, 1])
    #     visualizer.add_boxes(corners_target.reshape(-1, 3))
    #     vispy.app.run()

    iou = np.zeros((pred.shape[0], target.shape[0]))
    for p, c_pred in enumerate(corners_pred):
        for t, c_target in enumerate(corners_target):
            # if DEBUG and t == 4 and p == 47:
            #     print(f'IOU: {calc_iou(c_pred, c_target)}')
                # visualizer = Visualizer()
                # visualizer.add_boxes(c_pred, color=[1, 0, 0, 1])
                # visualizer.add_boxes(c_target)
                # vispy.app.run()

            iou[p, t] = calc_iou(c_pred, c_target)

    return iou


def compute_recall(pred, target, threshold):
    '''
    Task 1
    input
        pred (N,7) proposed 3D bounding box labels
        target (M,7) ground truth 3D bounding box labels
        threshold (float) threshold for positive samples
    output
        recall (float) recall for the scene
    '''
    ious = get_iou(pred, target)  # rows=predicitons, columns=targets
    ious_thresh = ious >= threshold
    ious_sum = np.sum(ious_thresh, axis=0)
    recall = np.count_nonzero(ious_sum)/len(ious_sum) # sum the target columns -> Number of correct predicitons per target
    return recall
