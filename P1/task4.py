"""
Remove Motion Distortion due to movement of the car.
"""
import cv2
import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation

import data_utils

USE_IMAGE_TIMESTAMP = False

if len(sys.argv) > 1:
    FRAME = sys.argv[1]
    SHOW_IMAGES = False
else:
    FRAME = 37
    SHOW_IMAGES = True

class Transformer:
    tfs = {}

    def set_transform(self, name, rotation=np.diag([1, 1, 1]), translation=np.ones((3, 1))):
        tf = self.tf_from_rot_trans(rotation, translation)
        self.tfs[name] = tf

    def get_rotation(self, name):
        return self.tfs[name][:3, :3]

    def get_translation(self, name):
        return self.tfs[name][:3, 4]

    @staticmethod
    def tf_from_rot_trans(rotation, translation):
        return np.vstack((np.hstack((rotation, translation)), [0, 0, 0, 1]))

def load_image(frame):
    '''
    Parameters:
    frame: Frame to load from dataset of problem 4
    '''
    image_name = str(frame).zfill(10) + '.png'
    image_path = os.path.join('data/problem_4/image_02/data', image_name)
    return cv2.imread(image_path)

def save_image(image, prefix=''):
    '''
    Parameters:
    image: image to save
    prefix: image is saved as prefix + framenumber.png

    Saves image to /gen_images/.
    '''
    if not os.path.exists('gen_images'):
        os.makedirs('gen_images')

    image_name = prefix + str(FRAME).zfill(10) + '.png'
    cv2.imwrite('gen_images/' + image_name, image)

def load_velo_data(frame):
    '''
    frame: Frame for which the Velodyne points should be loaded
    return: Velodyne points as np.array
    '''
    data_name = str(frame).zfill(10) + '.bin'
    data_path = os.path.join('data/problem_4/velodyne_points/data', data_name)

    return data_utils.load_from_bin(data_path)

def load_oxts_data(frame):
    '''
    frame: Frame for which the oxts data should be loaded
    return: velocity, angular_rate
    '''
    data_name = str(frame).zfill(10) + '.txt'
    data_path = os.path.join('data/problem_4/oxts/data', data_name)
    return [data_utils.load_oxts_velocity(data_path), data_utils.load_oxts_angular_rate(data_path)]


def load_timestamps(frame):
    '''
    frame: Frame for which the timestamps should beloaded
    return: timestampt dict with all the timestamps.
    '''
    timestamps = {}

    timestamps['velo_start'] = data_utils.compute_timestamps('data/problem_4/velodyne_points/timestamps_start.txt',
                                                             frame)
    timestamps['velo_forward'] = data_utils.compute_timestamps('data/problem_4/velodyne_points/timestamps.txt', frame)
    timestamps['velo_end'] = data_utils.compute_timestamps('data/problem_4/velodyne_points/timestamps_end.txt', frame)

    timestamps['oxts'] = data_utils.compute_timestamps('data/problem_4/oxts/timestamps.txt', frame)
    timestamps['image'] = data_utils.compute_timestamps('data/problem_4/image_02/timestamps.txt', frame)

    timestamps['image2'] = timestamps['image'] - 7e-3

    return timestamps

def filter_velodyne_data(velo_points, angles, fov=np.pi):
    '''
    velo_points: Velodyne points to filter
    angles: Angles corresponding to the points
    fov: total fov that should remain after operation
    Removes all points that are not within field of view.
    return: fitlered Velodyne points
    '''
    mask = (angles > -fov/2) & (angles < fov/2)
    return velo_points[mask], angles[mask]

def calculate_angle(velo_points):
    '''
    velo_points: Velodyne points to calculate angles for
    Calculate angle
    return: angle for each Velodyne point
    '''
    return np.arctan2(velo_points[:, 1], velo_points[:, 0])

def project_points(points, proj_image_velo, image):
    '''
    points: points to project
    proj_image_velo: projection matrix to use
    image: image to project points onto
    return: image with points projected onto
    '''
    # points = filter_velodyne_data(points)
    projected_points = np.transpose(np.matmul(proj_image_velo, np.transpose(points[:, :4])))

    projected_points[:, 0] /= projected_points[:, 2]
    projected_points[:, 1] /= projected_points[:, 2]

    distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
    colors = data_utils.depth_color(distances, max_d=max(distances), min_d=min(distances))

    image_proj = data_utils.print_projection_plt(points=projected_points.transpose(), color=colors, image=image)

    return image_proj

def load_transformer():
    '''
    Returns transformer object with all the loaded transforms needed for task 4.
    '''
    transformer = Transformer()

    R_cam2_velo, T_cam2_velo = data_utils.calib_velo2cam('data/problem_4/calib_velo_to_cam.txt')
    transformer.set_transform('cam_velo', rotation=R_cam2_velo, translation=T_cam2_velo)

    transformer.tfs['image_cam'] = data_utils.calib_cam2cam('data/problem_4/calib_cam_to_cam.txt', mode='02')

    # we can use velo2cam again, since the calib_xxx.txt files have the same format
    R_velo_imu, T_velo_imu = data_utils.calib_velo2cam('data/problem_4/calib_imu_to_velo.txt')
    transformer.set_transform('velo_imu', rotation=R_velo_imu, translation=T_velo_imu)

    return transformer

if __name__ == '__main__':
    # Import data
    velo_points = load_velo_data(FRAME)
    angles = calculate_angle(velo_points)
    # only consider points facing forward. Doing this here could lead to
    # problems for extremeley high velocities/angular rates.
    # But for the given data, it saves processing time.
    velo_points, angles = filter_velodyne_data(velo_points, angles, fov=100/180 * np.pi)

    timestamps = load_timestamps(FRAME)  # dictionary that stores the different timestamps
    image = load_image(FRAME)
    transformer = load_transformer()  # object that stores all the transforms
    # Naming convention: from second frame to first frame
    transformer.tfs['image_velo'] = transformer.tfs['image_cam'] @ transformer.tfs['cam_velo']

    oxts_velocity, oxts_anglular_rate = load_oxts_data(FRAME)
    R_velo_imu = transformer.get_rotation('velo_imu')

    # transform imu data to velodyne frame
    velo_velocity = R_velo_imu @ oxts_velocity
    velo_anglular_rate = R_velo_imu @ oxts_anglular_rate

    # duration of one complete rotation of velodyne
    rotation_time = (timestamps['velo_end'] - timestamps['velo_start'])
    time_per_rad = rotation_time / (2 * np.pi)
    time_image_fw = timestamps['image'] - timestamps['velo_forward']

    angle_start = (timestamps['velo_forward'] - timestamps['velo_start'])/time_per_rad

    # if angle_start is bigger than pi, it's on the right half and should be negative
    if angle_start > np.pi:
        angle_start = angle_start - 2*np.pi

    # transform to homogenous coordinates
    velo_points = np.pad(velo_points, pad_width=((0, 0), (0, 1)), mode='constant', constant_values=1)

    ### correct points. The actual task of the exercise
    points_corrected = np.zeros(velo_points.shape)
    for i, point in enumerate(velo_points):
        angle = angles[i]

        # correct for the case that the lidar starts scan in front half, but never happens for given data
        if angle < angle_start < 0:
            angle = angle + 2*np.pi
        if angle > angle_start > 0:
            angle = angle - 2*np.pi

        if not USE_IMAGE_TIMESTAMP:
            # calculate time since the velodyne was facing forward. This is negative if the velodyne faces forward in the future
            time_since_image = - angle * time_per_rad
        else:
            # calculate time since the image was taken
            time_since_image = - (time_image_fw + angle*time_per_rad)

        # car rotation in the mean time
        car_rot = time_since_image * velo_anglular_rate[2]
        R = Rotation.from_euler('z', car_rot).as_matrix()
        # car translation in the mean time
        car_trans = time_since_image * velo_velocity
        tf_corrected_velo = transformer.tf_from_rot_trans(R, car_trans.reshape(-1, 1))
        points_corrected[i] = tf_corrected_velo @ point

    # project points onto image
    image_org = project_points(points=velo_points, proj_image_velo=transformer.tfs['image_velo'], image=image)
    image_corrected = project_points(points=points_corrected, proj_image_velo=transformer.tfs['image_velo'],
                                     image=image)

    if SHOW_IMAGES:
        cv2.imshow("original", image_org)
        cv2.imshow("Corrected with Velo timestamp", image_corrected)
        cv2.waitKey(0)


    save_image(image_corrected)
    save_image(image_org, 'naive_')

    print('Task 4 done on frame ' + str(FRAME) + '.')
