import cython
import numpy as np
from numpy import cos, sin
from timeit import default_timer as timer

def rectangle_corners(preds, delta):
    """
    Computes the bottom corner coordinates for a bboxs
    input
        preds (N,7) 3D bounding box with (x, y, z, h, w, l, ry)
        delta, int by which the box size is increased in each direction
    output
        corners (N, 4, 3) corner coordinates in the rectified reference frame
        y:   (N) np array, stores the z coordinates of the bottom corners
        y+h: (N) np array, stores the z coordinates of the top corners
    corner order:
        1 -------- 0
       /|         /|
      2 -------- 3 .
      | |        | |
      . 5 -------- 4
      |/         |/
      6 -------- 7
    """
    x, y, z, h, w, l, r = np.hsplit(preds, preds.shape[1])
    r = -r + np.pi/2
    dx = w/2
    dz = l/2
    dxcos = dx*cos(r)
    dxsin = dx*sin(r)
    dzcos = dz*cos(r)
    dzsin = dz*sin(r)

    # Allocate memory for the corners coordinates. 
    rect_corners = np.zeros((preds.shape[0], 4, 2), dtype=np.float32)
    rect_corners[:, 0, :] = np.hstack([x + dxcos - dzsin, z + dzcos + dxsin]) # corner 4
    rect_corners[:, 1, :] = np.hstack([x - dxcos - dzsin, z + dzcos - dxsin]) # corner 5
    rect_corners[:, 2, :] = np.hstack([x + dxcos + dzsin, z - dzcos + dxsin]) # corner 7
    # rect_corners[:, 3, :] = np.hstack([x - dxcos + dzsin, z - dzcos - dxsin]) # corner 7

    # Adjust box height according to instruction sheet. 
    yh = y + delta
    yl = yh - h
    return rect_corners, yl.flatten(), yh.flatten()

@cython.boundscheck(False)
@cython.wraparound(False)
def assign_points_to_rectanlges(xyz, corners, z_bottom, z_top, rvec45, rvec47, rect_side_length, max_points):
    """
    Assign every point of the point cloud to all bboxes it is in w.r.t. its coordinates. 
    N  ... no. of points of the lidar point cloud
    K  ... no. of predicted bboxes
    K' ... no. of relevant bboxes, i.e. ones that have actual points located inside them
    input
        box:      (K, 7) np array, 7 bbox features
        xyz:      (N, 3) np array, (x, y, z) coordinates of a point
        features: (N, 128) np array, 128 features per point
        valid_pred:  (K', 7) np array, 7 bbox features
        pooled_xyz:  (K', 512, 3) np array, 512 points, 3 (x, y, z) coordinates of a point
        pooled_feat: (K', 512, 128) np array, 128 features per point
        corners:     (K, 4, 2) np array, 4 corners, 2 (x, y) coordinates
        z_bottom:    (K) np array, stores bottom z coordinate for every point
        z_top:       (K) np array, stores top z coordinate for every point
        rvec45:   (K, 2) np array, 2 (x, y lengths of rectangle side 45)
        rvec47:   (K, 2) np array, 2 (x, y lengths of rectangle side 56)
        rect_side_length: (K, 2) np array, 2 dotproduct of vector 45 and vector 56 with itselfes
        max_entries: int, max. no. of points assigned per bbox. 
    output
        The relevant data is stored in valid_pred, pooled_xyz, and pooled feat (pass by reference)
    """
    pntsxy = np.transpose(np.array([xyz[:, 0], xyz[:, 2]], dtype=np.float32))
    pntsz = np.transpose(np.array(xyz[:, 1], dtype=np.float32))
    # Index storage. 
    point_idxs, box_idxs = [], []

    # Loop over predicted bboxes. 
    for c_idx, c in enumerate(corners):
        # Extract indeces of points that lie within the bbox. The following 3 conditions must be satisfied.
        idxs = (pntsz >= z_bottom[c_idx]) & (pntsz <= z_top[c_idx]) # Height.
        qv = pntsxy - c[0] # Query vector.
        # Projections onto rectangle side 45.
        pjs45 = np.dot(qv, rvec45[c_idx])
        idxs &= np.logical_and(0 <= pjs45, pjs45 <= rect_side_length[c_idx, 0])
        # Projections onto rectangle side 47.
        pjs47 = np.dot(qv, rvec47[c_idx])
        idxs &= (0 <= pjs47) & (pjs47 <= rect_side_length[c_idx, 1])
        idxs = np.where(idxs)[0]
        # Sample points for non-empty boxes with less than max_entries entries.
        idxs_len = idxs.shape[0]
        num_to_sample = max_points - idxs_len
        if idxs_len > 0:
            if num_to_sample < 0:
                # Randomly sample until 512
                point_idxs.append(idxs[np.random.randint(max_points, size=max_points)])
            elif num_to_sample > 0:
                # fill with random samples until 512
                point_idxs.append(np.concatenate((idxs, idxs[np.random.randint(idxs_len, size=max_points-idxs_len)])))
            else:
                # exactly 512
                point_idxs.append(idxs)
            box_idxs.append(c_idx)

    return point_idxs, np.asarray(box_idxs)

@cython.boundscheck(False)
@cython.wraparound(False)
def roi_pool(pred, xyz, feat, config):
    '''
    Task 2
    a. Enlarge predicted 3D bounding boxes by delta=1.0 meters in all directions.
        As our inputs consist of coarse detection results from the stage-1 network,
        the second stage will benefit from the knowledge of surrounding points to
        better refine the initial prediction.
    b. Form ROI's by finding all points and their corresponding features that lie 
        in each enlarged bounding box. Each ROI should contain exactly 512 points.
        If there are more points within a bounding box, randomly sample until 512.
        If there are less points within a bounding box, randomly repeat points until
        512. If there are no points within a bounding box, the box should be discarded.
    input
        pred (N,7) bounding box labels; (N, x, y, z, h, w, l, r)
        xyz (N,3) point cloud
        feat (N,C) features
        config (dict) data config
    output
        valid_pred (K', 7)
        pooled_xyz (K', M, 3)
        pooled_feat (K', M, C)
            with K' indicating the number of valid bounding boxes that contain at least
            one point
    useful config hyperparameters
        config['delta'] extend the bounding box by delta on all sides (in meters)
        config['max_points'] number of points in the final sampled ROI
    '''
    # Enlarge bboxes.
    delta = config['delta']*2
    cpred = np.copy(pred)
    cpred[:, 3:6]+=delta

    # Ccorners of bounding boxes. 
    corners, z_bottom, z_top = rectangle_corners(cpred, config['delta'])

    # Vectors between corners. 
    vec45 = corners[:, 1] - corners[:, 0] # Corner 45.
    vec47 = corners[:, 2] - corners[:, 0] # Corner 47.
    # Side length vector products. 
    rsides = np.zeros((vec45.shape[0], 2), dtype=np.float32)
    rsides[:, 0] = np.sum(np.multiply(vec45, vec45), axis=1, keepdims=True).flatten()
    rsides[:, 1] = np.sum(np.multiply(vec47, vec47), axis=1, keepdims=True).flatten()

    # Get indexes of points inside bboxes. 
    idxs, box_idxs = assign_points_to_rectanlges(xyz, corners, z_bottom, z_top,
                                                 vec45, vec47, rsides, config['max_points'])
    # Extract data using the indexes. 
    pooled_xyz = xyz[idxs]
    pooled_feat = feat[idxs]
    valid_pred = pred[box_idxs]

    # for i in range(valid_pred.shape[0]):
    #     plt.scatter(pooled_xyz[i][:, 0], pooled_xyz[i][:, 2], c='b')
    #     for corner in corners[i]:
    #         plt.scatter(corner[0], corner[1], c='r')
    #     plt.savefig(f'./plots/dots_plot_{i}.png')
    #     plt.close()
     
    return valid_pred, pooled_xyz, pooled_feat
