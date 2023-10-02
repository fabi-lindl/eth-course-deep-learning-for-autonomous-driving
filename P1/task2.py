"""
Creates a laser identification of the point cloud and 
maps it to the camera 2.
"""
import os
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data

# Import data. 
data_path = os.path.join('./data', 'data.p')
# Extract lidar point cloud. 
data = load_data(data_path)
pntcloud = data['velodyne']

# Show data. 
# print(data.keys())

# Change color of color_map from BGR to RGB.
for i in data['color_map'].values():
    i[0], i[2] = i[2], i[0]

# Create image from given array data.
# Data type needs to be converted into an uint8 (see pillow doc). 
im = Image.fromarray(data['image_2'].astype(np.uint8))

# Extract lidar points of a certain angle.
# Only extract data necessary for the camera view field to store less
# data and reduce computation time. 
pa = 40*np.pi/180 # 40° chose. 
# Store colors corresponding to the extracted lidar points. 
colors = []
# Relevant points for camera view angle. 
pts = np.array([0, 0, 0, 0])
cnt = 0
for pnt in pntcloud:
    # Select only points in the forward car direction (lidar cs: positve x-axis). 
    if (pnt[0] > 0):
        # Check if angle is within the selected view field. 
        angle = np.arctan(pnt[1]/pnt[0])
        if angle < pa and angle > -pa:
            pts = np.vstack([pts, pnt])
            colors.append(data['sem_label'][cnt][0])
    cnt+=1
pts = np.delete(pts, 0, 0)

# Set the last lidar point vector entry to one to init
# the vector set-up for the hom-coordinate transformations. 
ones = np.ones(len(pts[:, 0]))
pts[:, -1] = ones

# Compute projection matrix. 
# Projection matrix consists of two steps. 
# 1) Transformation of the lidar cs to cam0 zero cs.
# 2) Transformation of the cam0 cs to cam2 image plane. 
tm = np.matmul(data['P_rect_20'], data['T_cam0_velo'])
# Transform point cloud onto cam2 image plane. 
pts = np.transpose(np.matmul(tm, np.transpose(pts)))
# Normalize the x and y coordinates, i.e. divide by the hom. z-value. 
pts[:, 0]/=pts[:, 2]
pts[:, 1]/=pts[:, 2]

# Add points to the image. 
# Change image to an ImageDraw object for annotations in the image. 
imdraw = ImageDraw.Draw(im)
cnt = 0
for pnt in pts:
    imdraw.point([pnt[0], pnt[1]], tuple(data['color_map'][colors[cnt]]))
    cnt+=1

# Store image in current directory. 
im.save('image_labeled_lidar_points.png')
# Display image with projected and labelled lidar points. 
im.show()




# Compute 3d bboxes for cars.
numObjs = len(data['objects'])
# 8 edge points with 3 coordinates (x, y, z) each. 
# Create [numObjsx8x3] array to store the bbox edge points. 
objbbx = np.zeros(numObjs*8*3).reshape(numObjs, 8, 3)
idx = 0
# Loop over all objects and compute the bbox coordinate points
# projected onto the camera image plane. 
for obj in data['objects']:
    # Create storage array [4x8] for edge point coordinates. 
    bboxEdges = np.zeros(32).reshape(4, 8)
    # Extract data from object data format. 
    h = obj[8]     # height
    w = obj[9]     # width
    l = obj[10]    # length
    # Add 90 degrees to the given rotation to transform the
    # roation into the 2d coordinate system used for computing 
    # the edge positions in the x, z plane.
    yrot = np.pi/2 + obj[-2] # Rotation around y axis.
    # Bbox edges.
    xd = w/2 # x-distance (lidar cs)
    zd = l/2 # z-distance (lidar cs)
    # Set up rotation matrix in the 2d camera x-, z-plane. 
    rot2d = np.array([
        [np.cos(yrot), -np.sin(yrot)],
        [np.sin(yrot), np.cos(yrot)]]
    )
    cnt = 0
    for i in range(1, -2, -2):     # x coordinate
        for j in range(1, -2, -2): # z coordinate
            if i < 0:
                # Required for the edge point ordering. Otherwise bboxes are drawn incorrectly. 
                # Necessary for the pillow polygon function. 
                j *= -1
            # Point coordinates in the x-, z-plane. 
            ov = np.array([xd*i, zd*j])
            ov = np.matmul(np.transpose(rot2d), ov) # Rotation. 
            x = obj[11] + ov[0] # Object center plus offset. 
            z = obj[13] + ov[1] # Object center plus offset. 
            # Store computed edge point coordinates. 
            bboxEdges[:, cnt] = np.array([x, obj[12], z, 1])     # bottom point
            bboxEdges[:, cnt+4] = np.array([x, obj[12]-h, z, 1]) # top point
            cnt+=1
    
    # Project 3d cam0 coordinates onto the image plane of any other camera. 
    bbcam = np.matmul(data['P_rect_20'], bboxEdges)
    # Normalize the x and y coordinates, i.e. divide by the hom. z value. 
    bbcam[0, :]/=bbcam[2, :]
    bbcam[1, :]/=bbcam[2, :]
    # Store the transposed version of the matrix. This sort of data storge makes
    # drawing the bboxes in the image easier. 
    objbbx[idx] = np.transpose(bbcam)
    idx+=1

# Draw car bboxes on image. 
bboxdolor = (0, 255, 0)
for obj in objbbx:
    # Bottom polygon.
    x = np.vstack([obj[:4, 0], obj[:4, 1]])
    x = np.reshape(x, 8, order='F')
    imdraw.polygon(list(x), None, bboxdolor)
    # Top polygon. 
    x = np.vstack([obj[4:, 0], obj[4:, 1]])
    x = np.reshape(x, 8, order='F')
    imdraw.polygon(list(x), None, bboxdolor)
    # Vertical lines
    for i in range(0, 4):
        vl = np.hstack([obj[i, :2], obj[i+4, :2]])
        imdraw.line(list(vl), bboxdolor)

# Store image in current directory. 
im.save('image_labeled_lidar_points_bboxes.png')
# Display image with labelled lidar points. 
im.show()
