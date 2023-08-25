# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch
import zlib
import matplotlib.pyplot as plt
import open3d as o3d
import math

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools

PREVENT_PCL = False
PREVENT_SHOW_CHANNELS = False
OUTPUT_PCL_PNG = False
Frame = 50

# visualize lidar point-cloud
def show_pcl(pcl):

    if(PREVENT_PCL):
        return
    ####### ID_S1_EX2 START #######     
    #######
    print("student task ID_S1_EX2")
    data = pcl[:,:3]

    # step 1 : initialize open3d with key callback and create window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # step 2 : create instance of open3d point-cloud class
    pcd = o3d.geometry.PointCloud()
    
    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
    pcd.points = o3d.utility.Vector3dVector(data)
    
    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    vis.add_geometry(pcd)

    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)
    vis.run()
    
    if OUTPUT_PCL_PNG:
        global Frame
        vis.capture_screen_image(f'./output/pcl-{Frame:03d}.png')
        Frame += 1

    #######
    ####### ID_S1_EX2 END #######     
       

# visualize range image
def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######     
    #######
    print("student task ID_S1_EX1")

    # step 1 : extract lidar data and range image for the roof-mounted lidar
    calib_lidar = [obj for obj in frame.context.laser_calibrations if obj.name == lidar_name][0]
    extrinsic = np.array(calib_lidar.extrinsic.transform).reshape(4,4)
    # Azimuth correction is not needed, but let's leave it here just in case
    az_correction = math.atan2(extrinsic[1,0], extrinsic[0,0])
    
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0]
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0: # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    
    # step 2 : extract the range and the intensity channel from the range image
    ri_range = ri[:,:,0]
    ri_intensity = ri[:,:,1]

    # step 2.5: take a ±90 deg crop
    image_cols = ri_range.shape[1]
    assert(image_cols == ri_intensity.shape[1])
    n_crop_cols_half = image_cols * 90 / 360
    ri_range = ri_range[:,int(image_cols/2-n_crop_cols_half):int(image_cols/2+n_crop_cols_half)]
    ri_intensity = ri_intensity[:,int(image_cols/2-n_crop_cols_half):int(image_cols/2+n_crop_cols_half)]
    
    # step 3 : set values <0 to zero
    ri[ri<0]=0.0
    
    # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    max_value = np.max(ri_range)
    min_value = np.min(ri_range)
    ri_range = ((ri_range - min_value) / (max_value-min_value)) * 255
    ri_range = ri_range.astype(np.uint8)
    # Make sure the range is optimally used
    assert np.min(ri_range) == 0
    assert np.max(ri_range) == 255
    
    # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    p99 = np.percentile(ri_intensity, 99)
    p1 = np.percentile(ri_intensity, 1)
    valid_values = ri_intensity[np.where(np.logical_and(ri_intensity<=p99, ri_intensity>=p1))]
    max_value = np.max(valid_values)
    min_value = np.min(valid_values)
    ri_intensity[ri_intensity<min_value]=min_value
    ri_intensity[ri_intensity>max_value]=max_value
    ri_intensity = ((ri_intensity - min_value) / (max_value-min_value)) * 255
    ri_intensity = ri_intensity.astype(np.uint8)
    # Make sure the range is optimally used
    assert np.min(ri_intensity) == 0
    assert np.max(ri_intensity) == 255

    # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
    img_range_intensity = np.vstack([ri_range, ri_intensity], dtype=np.uint8)
    
    #######
    ####### ID_S1_EX1 END #######     
    
    return img_range_intensity


# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0] 

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######     
    #######
    print("student task ID_S2_EX1")

    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    min_y = np.min(lidar_pcl[:,0])
    max_y = np.max(lidar_pcl[:,0])
    y_range = max_y - min_y
    min_x = np.min(lidar_pcl[:,1])
    max_x = np.max(lidar_pcl[:,1])
    x_range = max_x - min_x
    range = np.max([x_range, y_range])

    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates    
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:,0] = ((lidar_pcl_cpy[:,0] - min_y) / range) * configs.bev_height

    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    # y = 0 must be at the center of the image (range/2)
    lidar_pcl_cpy[:,1] = ((lidar_pcl_cpy[:,1] + range/2) / range) * configs.bev_width

    # Clamp intensity between 1st and 99th percentiles to remove outliers
    top_intensity = np.percentile(lidar_pcl_cpy[:, 3], 99)
    bottom_intensity = np.percentile(lidar_pcl_cpy[:, 3], 1)
    top_mask = np.where(lidar_pcl_cpy[:, 3]>=top_intensity)
    bottom_mask = np.where(lidar_pcl_cpy[:, 3]<=bottom_intensity)
    lidar_pcl_cpy[:, 3][top_mask] = top_intensity
    lidar_pcl_cpy[:, 3][bottom_mask] = bottom_intensity
    # Scale height channel by 8 (arbitrary) before casting to int32; otherwise, we loose a lot of precision
    # since most height values lie in the 1~2 meter range
    HEIGHT_MULTIPLIER = 8
    lidar_pcl_cpy[:, 2] = lidar_pcl_cpy[:, 2] * HEIGHT_MULTIPLIER
    lidar_pcl_cpy[:, 3] = lidar_pcl_cpy[:, 3] * 255
    lidar_pcl_cpy = lidar_pcl_cpy.astype(np.int32)
    
    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    show_pcl(lidar_pcl_cpy)

    #######
    ####### ID_S2_EX1 END #######     
    
    
    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######     
    #######
    print("student task ID_S2_EX2")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    # +1 because x,y coordinates will get to exactly bev_width, bev_height
    intensity_map = np.zeros([configs.bev_width+1, configs.bev_height+1])

    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    # (of all points with the same x,y pixel coordinates, we want to keep only the one with the greatest height)
    idx_sort_height = np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_by_height = lidar_pcl_cpy[idx_sort_height]

    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    _, idx_height_unique = np.unique(lidar_pcl_by_height[:, 0:2], axis=0, return_index=True)
    lidar_pcl_by_height = lidar_pcl_by_height[idx_height_unique]
    lidar_pcl_top = np.copy(lidar_pcl_by_height) # this before selecting uniques?

    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map 
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
    max_intensity = np.percentile(lidar_pcl_by_height[:, 3], 99)
    min_intensity = np.min(lidar_pcl_by_height[:, 3])
    # Fit intensity in the range [0-1]
    intensity_map[lidar_pcl_by_height[:,0], lidar_pcl_by_height[:,1]] = (lidar_pcl_by_height[:, 3] - min_intensity) / (max_intensity-min_intensity)#(lidar_pcl_by_height[:, 3]/HEIGHT_MULTIPLIER) * mean_value/7

    #cv2.imshow('intensity_channel', intensity_map)
    #cv2.waitKey()

    #######
    ####### ID_S2_EX2 END ####### 


    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######     
    #######
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    # +1 because x,y coordinates will get to exactly bev_width, bev_height
    height_map = np.zeros([configs.bev_width+1, configs.bev_height+1])

    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map 
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map
    # Actual height (m) was multiplied by HEIGHT_MULTIPLIER, so we multiply
    #  top_z and bottom_z to preserve resolution
    top_z = configs.lim_z[1] * HEIGHT_MULTIPLIER
    bottom_z = configs.lim_z[0] * HEIGHT_MULTIPLIER
    # bottom_z was already subtracted from height at the beginning of the function
    height_channel = (lidar_pcl_by_height[:,2]) / (top_z - bottom_z)
    height_map[lidar_pcl_by_height[:,0], lidar_pcl_by_height[:,1]] = height_channel

    ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    if not PREVENT_SHOW_CHANNELS:
        cv2.imshow('intensity_channel', intensity_map)
        cv2.waitKey()
        cv2.imshow('height_channel', height_map)
        cv2.waitKey()

    #######
    ####### ID_S2_EX3 END #######       

    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) 
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts
        
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps


