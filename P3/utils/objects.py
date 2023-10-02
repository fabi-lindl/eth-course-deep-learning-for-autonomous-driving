"""
This file contains all objects for the DLAD projects. 
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import copy

class Visualizer2D():

    def __init__(self):
        # Storage for the provided data dictionary. 
        self.data = {}

    def bgr_to_rgb_colors(self, scale=0):
        """
        Parameters:
        scale: Integer to decide if the color map should be scaled to values in [0, 1].
        -----------
        Transforms the given color map from BGR color codes to RGB codes. 
        If scale != 0 then a copy of the color map is generated, its values are scaled
        to the interval [0, 1] and it gets returned. 
        """
        if scale == 0:
            for i in self.data['color_map'].values():
                i[0], i[2] = i[2], i[0]
        else:
            colorMap = copy.deepcopy(self.data['color_map'])
            for i in colorMap.values():
                i[0], i[2] = i[2], i[0]
                i[0] /= 255
                i[1] /= 255
                i[2] /= 255
            return colorMap


    def birds_eye_view(self):
        """
        Parameters:
        lidarData: Contains all points of the lidar point cloud. 
                   2D numpy array  [[x y z intensity] ... ]
        -----------
        Creates a 2d plot birds-eye-view of the given lidar point cloud. 
        Stores the plot in the current directory as 'bev_of_lidar_point_cloud.png'.
        """
        # Find min and max values in x and y direction. 
        xmin = int(min(self.data['velodyne'][:, 0])-1)
        xmax = int(max(self.data['velodyne'][:, 0])+1)
        ymin = int(min(self.data['velodyne'][:, 1])-1)
        ymax = int(max(self.data['velodyne'][:, 1])+1)

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
        for pnt in self.data['velodyne']:
            xidx = xOrigin + int(pnt[0]/0.2)
            yidx = yOrigin + int(pnt[1]/0.2)
            # Store point of highest intensity in the bins. 
            if pnt[3] > z[yidx][xidx]:
                z[yidx][xidx] = pnt[3]

        # Store image in current directory. 
        # plt.savefig('bev_of_lidar_point_cloud.png')
        # Plot a greyscale map of the lidar point cloud. 
        plt.imshow(z, cmap='gray', vmin=0, vmax=1, origin='lower') # Enable to show plot. 
        plt.show()


    def extract_lidar_points_of_view_angle(self, viewAngle):
        """
        Parameters:
        viewAngle: Integer that provides the angle within the points should be extracted. 
        colors   : Integer that defines if the colors of the extracted lidar points should be returned. 
        -----------
        Extracts all points from the lidar point cloud that are within the given view angle. 
        Return these points in the lidar cloud format and returns a list of color codes for
        each lidar point for the semantic labeling. 
        """
        # Extract lidar points of a certain angle.
        # Perception angle. 
        # Only extract data necessary for the camera view field to store less
        # data and reduce computation time. 
        pa = viewAngle*np.pi/180
        # Store colors corresponding to the extracted lidar points. 
        colors = []
        pnts = np.array([0, 0, 0, 0])
        cnt = 0
        for i in self.data['velodyne']:
            # Select only points in the forward car direction (lidar cs: positve x-axis). 
            if (i[0] > 0):
                # Check if angle is within the selected view field. 
                angle = np.arctan(i[1]/i[0])
                if angle < pa and angle > -pa:
                    pnts = np.vstack([pnts, i])
                    colors.append(self.data['sem_label'][cnt][0])
            cnt+=1
        pnts = np.delete(pnts, 0, 0)
        return pnts, colors
    

    def lidar_ray_ids(self, tmCamxx, tmCamxLidar, imageKey='image_2', camAngle=40):
        """
        Parameters:  
        tmCamxx    : Transformation matrix from one cam coordinate sys. to a camera image plane.
        tmCamxLidar: Transformation matrix from lidar to a camera.
        imageKey   : String as key to the given image data. Default is the data of cam2.  
        camAngle   : Angle of the camera view field. 
        -----------
        Maps lidar points into the given image and colors them according to the laser id. 
        Stores a plot in the current directory as 'lidar_id_mapping.png'
        """
        # Extract relevant lidar points for camera view. 
        pnts, colors = self.extract_lidar_points_of_view_angle(camAngle)
        
        # Set the last lidar point vector entry to one to init
        # the vector set-up for the hom-coordinate transformations. 
        ones = np.ones(len(pnts[:, 0]))
        pnts[:, -1] = ones

        # Compute projection matrix.  
        tm = np.matmul(tmCamxx, tmCamxLidar)
        # Transform point cloud onto cam2 image plane. 
        pnts = np.transpose(np.matmul(tm, np.transpose(pnts)))
        # Normalize the x and y coordinates, i.e. divide by the hom. z value. 
        pnts[:, 0]/=pnts[:, 2]
        pnts[:, 1]/=pnts[:, 2]

        # Add points to the image. 
        image = Image.fromarray(self.data[imageKey].astype(np.uint8))
        # Change image to an ImageDraw object for annotations in the image. 
        imdraw = ImageDraw.Draw(image)

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
        image.save('lidar_id_mapping.png')
        # Display image with projected and labelled lidar points.
        image.show() # Enable if you want the image shown. 


    def semantic_labeling(self, tmCamxx, tmCamxLidar, imageKey='image_2', camAngle=40, save=1):
        """
        Parameters: 
        tmCamxx    : Transformation matrix from one cam coordinate sys. to a camera image plane.
        tmCamxLidar: Transformation matrix from lidar to a camera.
        imageKey   : String as key to the given image data. Default is the data of cam2.
        camAngle   : Angle of the camera view field. 
        save       : Int to decide if the image should be saved. 
        -----------
        Maps lidar points onto a given image and colors them according to the semantics.
        If save==1 an image is stored in the current directory (image_labeled_lidar_points.png).
        """
        # Extract relevant lidar points for camera view. 
        pnts, colors = self.extract_lidar_points_of_view_angle(camAngle)

        # Compute projection matrix.
        tm = np.matmul(tmCamxx, tmCamxLidar)
        # Transform point cloud onto cam image plane. 
        pnts = np.transpose(np.matmul(tm, np.transpose(pnts)))
        # Normalize the x and y coordinates, i.e. divide by the hom. z-value. 
        pnts[:, 0]/=pnts[:, 2]
        pnts[:, 1]/=pnts[:, 2]

        # Add points to the image. 
        # Change image to an ImageDraw object for annotations in the image.
        image = Image.fromarray(self.data[imageKey].astype(np.uint8))
        imdraw = ImageDraw.Draw(image)
        cnt = 0
        for pnt in pnts:
            imdraw.point([pnt[0], pnt[1]], tuple(self.data['color_map'][colors[cnt]]))
            cnt+=1
        
        # Store and show image result. 
        if save == 1:
            # Store image in current directory. 
            image.save('image_labeled_lidar_points.png')
            # Display image with projected and labelled lidar points. 
            image.show() # Enable i you want to show the image. 
        else:
            return image


    def bbox_coordinates(self, cs, tm):
        """
        Parameters:
        cs: If cs == 'cam' the bbox coordinates of the camera image plane are returned.
            Else the 3d coordinates are returned with respect to the lidar cs. 
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
        if cs == 'cam':
            check = True
        else:
            check = False
        # Compute 3d bboxes for cars.
        numObjs = len(self.data['objects'])
        # 8 edge points with 3 coordinates (x, y, z) each. 
        # Create [numObjsx8x3] array to store the bbox edge points. 
        objbboxes = np.zeros(numObjs*8*3).reshape(numObjs, 8, 3)
        idx = 0
        # Loop over all objects and compute the bbox coordinate points
        # projected onto the camera image plane. 
        for obj in self.data['objects']:
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
            
            # Project edge coordinates into cam or lidar coordinate system. 
            if check: # cam
                # Project 3d cam0 coordinates onto the image plane of any other camera. 
                bbcam = np.matmul(tm, bboxEdges)
                # Normalize the x and y coordinates, i.e. divide by the hom. z value. 
                bbcam[0, :]/=bbcam[2, :]
                bbcam[1, :]/=bbcam[2, :]
                # Store the transposed version of the matrix. This sort of data storge makes
                # drawing the bboxes in the image easier. 
                objbboxes[idx] = np.transpose(bbcam)
            else: # lidar
                bblidar = np.matmul(tm, bboxEdges)
                bblidar = np.delete(bblidar, 3, 0)
                bblidar = np.transpose(bblidar)
                objbboxes[idx] = bblidar
            idx+=1
        
        return objbboxes


    def semantic_labeling_bboxes(self, tmCamxx, tmCamxLidar, camAngle=40):
        """
        Parameters:
        tmCamxx    : Transformation matrix from one cam coordinate sys. to a camera image plane.
        tmCamxLidar: Transformation matrix from lidar to a camera.
        imageKey   : String as key to the given image data. Default is the data of cam2.
        camAngle  : Angle of the camera view field.  
        -----------
        Maps lidar points onto a given image, colors them according to the semantics and 
        draws 3d bboxes around car objects. 
        """
        # Compute the coordinates of the object bboxes. 
        objbboxs = self.bbox_coordinates('cam', tmCamxx)

        # Draw car bboxes on image. 
        # Create image with semantic point cloud labels and draw object. 
        image = self.semantic_labeling(tmCamxx, tmCamxLidar, save=0)
        imdraw = ImageDraw.Draw(image)
        bboxdolor = (0, 255, 0)
        for obj in objbboxs:
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
        image.save('image_labeled_lidar_points_bboxes.png')
        # Display image with labelled lidar points. 
        image.show() # Enable if you want to show the image.
