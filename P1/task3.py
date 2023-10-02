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

# Change color of color_map from BGR to RGB.
for i in data['color_map'].values():
    i[0], i[2] = i[2], i[0]

# From image from given array data.
# Data type needs to be converted into an uint8 (see pillow doc.). 
im = Image.fromarray(data['image_2'].astype(np.uint8))

# Extract lidar points of a certain angle.
# Perception angle. 
# Only extract data necessary for the camera view field to store less
# data and reduce computation time. 
pa = 40*np.pi/180
# Store colors corresponding to the extracted lidar points. 
colors = []
pnts = np.array([0, 0, 0, 0])
cnt = 0
for i in pntcloud:
    # Select only points in the forward car direction (lidar cs: positve x-axis). 
    if (i[0] > 0):
        # Check if angle is within the selected view field. 
        angle = np.arctan(i[1]/i[0])
        if angle < pa and angle > -pa:
            pnts = np.vstack([pnts, i])
            colors.append(data['sem_label'][cnt][0])
    cnt+=1
pnts = np.delete(pnts, 0, 0)

# Set the last lidar point vector entry to one to init
# the vector set-up for the hom-coordinate transformations. 
ones = np.ones(len(pnts[:, 0]))
pnts[:, -1] = ones

# Compute projection matrix. 
# Projection matrix consists of two steps. 
# 1) Transformation of the lidar cs to cam0 zero cs.
# 2) Transformation of the cam0 cs to cam2 image plane. 
tm = np.matmul(data['P_rect_20'], data['T_cam0_velo'])
# Transform point cloud onto cam2 image plane. 
pnts = np.transpose(np.matmul(tm, np.transpose(pnts)))
# Normalize the x and y coordinates, i.e. divide by the hom. z value. 
pnts[:, 0]/=pnts[:, 2]
pnts[:, 1]/=pnts[:, 2]

# Add points to the image. 
# Change image to an ImageDraw obje
# ct for annotations in the image. 
imdraw = ImageDraw.Draw(im)







# Create color dictionary for the 4 different lidar laser id colors. 
colorDict = {
    0:(255, 0, 0),
    1:(0, 255, 0),
    2:(0, 0, 255),
    3:(255, 153, 51)
}
# Init. the values of the colorTuple, which is used as color argument 
# for the draw function. 
colorTuple = colorDict[0]

# Variables for the following for loop. 
colorChange = 0 # Track if the color changed in interval (1). 
cnt = 0         # Track the current loop iteration. 
cidx = 0        # Current colorDict index of active drawing color. 
yprev = 0       # y coordinate of previous iteration point. 
inside = 0      # Track if previous point was inside interval (1). 
# Loop over all relevant points of the lidar cloud. 
for i in pnts:
    # Interval (1): Centered around cam2.
    # Drawing color changes according to height difference of previous points. 
    if i[0] > 604 and i[0] < 636:
        # Height difference between current point and previous point.     
        dy = abs(i[1]-yprev)
        # Compute the range of height difference of the last 10 points
        # of the current lidar id. Add std. to compensate for outliers. 
        minyPrev = min(pnts[cnt-10:cnt, 1])
        maxyPrev = max(pnts[cnt-10:cnt, 1])
        stdy = np.std(pnts[cnt-10:cnt, 1])
        dyprev = maxyPrev - minyPrev + stdy
        # Check drawing color change.
        # If delta y between the current point and the prev. point is 
        # bigger than dyprev, change drawing color. 
        if dy > dyprev:
            if cidx == 3:
                cidx = 0
            else:
                cidx+=1
            colorTuple = colorDict[cidx]
            # Change drawing color, new lidar id found.  
            colorChange = 1
        # Point is inside interval (1). 
        inside = 1
    # Both points outside of interval (1). 
    # If the current point is to the left of (1) and the previous point is to
    # the right of (1), the color must be changed. 
    elif i[0] < 605 and pnts[cnt-1][0] > 635:
        if cidx == 3:
            cidx = 0
        else:
            cidx+=1
        # Change drawing color, new lidar id found. 
        colorTuple = colorDict[cidx]
    # Current point to the right of interval (1) and previous piont inside of (1). 
    elif i[0] < 605 and inside == 1:
        # Check if color has already been changed in interval (1). 
        # If yes, don't change it again; it will skip one color id. 
        if colorChange == 0:
            if cidx == 3:
                cidx = 0
            else:
                cidx+=1
            # Change drawing color, new lidar id found. 
            colorTuple = colorDict[cidx]
        # Set inside to zero, the current point is the previous point for the 
        # next iteration, and it is outside of interval (1). 
        inside = 0

    # Draw point on the image. 
    imdraw.point([i[0], i[1]], colorTuple)
    # Update coordinate of previous iteration variable. 
    yprev = i[1]
    cnt+=1

# Store image in current directory.  
im.save('lidar_id_mapping.png')
# Display image with projected and labelled lidar points.
im.show()
