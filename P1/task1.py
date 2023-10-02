"""
Creates a 2D map of a Lidar point cloud
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data

# Import data. 
data_path = os.path.join('./data', 'data.p')
# Extract lidar point cloud. 
lidarData = load_data(data_path)['velodyne']

# Find min and max values in x and y direction. 
xmin = int(min(lidarData[:, 0])-1)
xmax = int(max(lidarData[:, 0])+1)
ymin = int(min(lidarData[:, 1])-1)
ymax = int(max(lidarData[:, 1])+1)

# Create grid for the 2D point cloud plot with accuracy of 0.2 m. 
# Storage bins for x and y direction.  
numBinsx = int((abs(xmin)+abs(xmax))/0.2)
numBinsy = int((abs(ymin)+abs(ymax))/0.2)

# Compute origin of coordinate system. 
xOrigin = abs(int(xmin/0.2))
yOrigin = abs(int(ymin/0.2))
# Create intensity array of grid size [y-dir x x-dir]. 
z = np.zeros((numBinsy, numBinsx))
# Fill z with intensity values of data.
for pnt in lidarData:
    xidx = xOrigin + int(pnt[0]/0.2)
    yidx = yOrigin + int(pnt[1]/0.2)
    # Store point of highest intensity in the bins. 
    if pnt[3] > z[yidx][xidx]:
        z[yidx][xidx] = pnt[3]

# Store image in current directory. 
plt.savefig('bev_of_lidar_point_cloud.png')
# Plot a greyscale map of the lidar point cloud. 
plt.imshow(z, cmap='gray', vmin=0, vmax=1, origin='lower')
plt.show()
