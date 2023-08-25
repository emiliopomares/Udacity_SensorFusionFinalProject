# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
from tools.objdet_models.darknet.utils.cal_intersection_rotated_boxes import PolyArea2D
from tools.objdet_models.darknet.utils.cal_intersection_rotated_boxes import intersection_area
import numpy as np
import matplotlib
#matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt

import torch
from shapely.geometry import Polygon
from operator import itemgetter

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# object detection tools and helper functions
import misc.objdet_tools as tools


# compute various performance measures to assess object detection
def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):
    
     # find best detection for each valid label 
    true_positives = 0 # no. of correctly detected objects
    center_devs = []
    ious = []
    n_detections = len(detections)
    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        if valid: # exclude all labels from statistics which are not considered valid

            # compute intersection over union (iou) and distance between centers

            ####### ID_S4_EX1 START #######     
            #######
            print("student task ID_S4_EX1 ")

            ## step 1 : extract the four corners of the current label bounding-box
            label_x, label_y, label_z, label_w, label_l, label_yaw = label.box.center_x, label.box.center_y, label.box.center_z, label.box.width, label.box.length, label.box.heading
            label_corners = tools.compute_box_corners(label_x,label_y,label_w,label_l,label_yaw)

            ## step 2 : loop over all detected objects
            for detection in detections:

                ## step 3 : extract the four corners of the current detection
                _, det_x, det_y, det_z, _, det_w, det_l, det_yaw = detection
                det_corners = tools.compute_box_corners(det_x,det_y,det_w,det_l,det_yaw)
                
                ## step 4 : computer the center distance between label and detection bounding-box in x, y, and z
                dist_x, dist_y, dist_z = det_x - label_x, det_y - label_y, det_z - label_z

                ## step 5 : compute the intersection over union (IOU) between label and detection bounding-box
                # Note: intersection_area had a bug apparently which made it report some iou even if the bboxes did not
                # overlap at all. A patch for that has been provided
                def compute_iou(box1, box2):
                    intersection = intersection_area(box1, box2)
                    union = PolyArea2D(box1) + PolyArea2D(box2) - intersection
                    return intersection/union
                
                iou = compute_iou(torch.tensor(det_corners), torch.tensor(label_corners))
                
                ## step 6 : if IOU exceeds min_iou threshold, store [iou,dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count
                if iou > min_iou:
                    matches_lab_det.append([iou, dist_x, dist_y, dist_z])
                    true_positives += 1
                
            #######
            ####### ID_S4_EX1 END #######     
        
        # find best match and compute metrics
        if matches_lab_det:
            best_match = max(matches_lab_det,key=itemgetter(1)) # retrieve entry with max iou in case of multiple candidates   
            ious.append(best_match[0])
            center_devs.append(best_match[1:])

    ####### ID_S4_EX2 START #######     
    #######
    print("student task ID_S4_EX2")
    
    # compute positives and negatives for precision/recall
    
    ## step 1 : compute the total number of positives present in the scene
    all_positives = 0
    for label, valid in zip(labels, labels_valid):
        if valid:
            all_positives += 1

    ## step 2 : compute the number of false negatives
    false_negatives = all_positives - true_positives
    
    ## step 3 : compute the number of false positives
    false_positives = n_detections - true_positives
    
    #######
    ####### ID_S4_EX2 END #######     
    
    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]
    #print(det_performance)
    
    return det_performance


# evaluate object detection performance based on all frames
def compute_performance_stats(det_performance_all, configs):

    # extract elements
    ious = []
    center_devs = []
    pos_negs = []
    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        pos_negs.append(item[2])
    
    ####### ID_S4_EX3 START #######     
    #######    
    print('student task ID_S4_EX3')

    ## step 1 : extract the total number of positives, true positives, false negatives and false positives
    total_all = 0
    tp_all = 0
    fn_all = 0
    fp_all = 0
    for performance in det_performance_all:
        values = performance[-1]
        total, tp, fn, fp = values
        total_all += total
        tp_all += tp
        fn_all += fn
        fp_all += fp

    ## step 2 : compute precision
    precision = tp_all / (tp_all + fp_all)

    ## step 3 : compute recall 
    recall = tp_all / (tp_all + fn_all)

    #######    
    ####### ID_S4_EX3 END #######     
    print('precision = ' + str(precision) + ", recall = " + str(recall) + ", f1 = " + str(2*precision*recall/(precision+recall)))   

    # serialize intersection-over-union and deviations in x,y,z
    ious_all = [element for tupl in ious for element in tupl]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    for tuple in center_devs:
        for elem in tuple:
            dev_x, dev_y, dev_z = elem
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)
    
    # compute statistics
    stdev__ious = np.std(ious_all)
    mean__ious = np.mean(ious_all)

    stdev__devx = np.std(devs_x_all)
    mean__devx = np.mean(devs_x_all)

    stdev__devy = np.std(devs_y_all)
    mean__devy = np.mean(devs_y_all)

    stdev__devz = np.std(devs_z_all)
    mean__devz = np.mean(devs_z_all)
    #std_dev_x = np.std(devs_x)
    
    # plot results
    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = ['detection precision', 'detection recall', 'intersection over union', 'position errors in X', 'position errors in Y', 'position error in Z']
    textboxes = ['', '', '',
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_x_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_x_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_y_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_y_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_z_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_z_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), )))]

    f, a = plt.subplots(2, 3)
    a = a.ravel()
    num_bins = 20
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for idx, ax in enumerate(a):
        ax.hist(data[idx], num_bins)
        ax.set_title(titles[idx])
        if textboxes[idx]:
            ax.text(0.05, 0.95, textboxes[idx], transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)          
    plt.tight_layout()
    plt.show()

