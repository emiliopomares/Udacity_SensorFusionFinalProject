# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Classes for track and track management
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
import collections
import math

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params

class Track:
    '''Track class with state, covariance, id, score'''
    def __init__(self, meas, id):
        print('creating track no.', id)
        print(f"meas: {meas.z}")
        M_rot = meas.sensor.sens_to_veh[0:3, 0:3] # rotation matrix from sensor to vehicle coordinates
        
        # Remove fixed initializations
        #self.x = np.matrix([[49.53980697],
        #                [ 3.41006279],
        #                [ 0.91790581],
        #                [ 0.        ],
        #                [ 0.        ],
        #                [ 0.        ]])
        #self.P = np.matrix([[9.0e-02, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
        #                [0.0e+00, 9.0e-02, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
        #                [0.0e+00, 0.0e+00, 6.4e-03, 0.0e+00, 0.0e+00, 0.0e+00],
        #                [0.0e+00, 0.0e+00, 0.0e+00, 2.5e+03, 0.0e+00, 0.0e+00],
        #                [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 2.5e+03, 0.0e+00],
        #                [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 2.5e+01]])
        #self.state = 'confirmed'
        #self.score = 0
        
        measure_z = np.ones((4,1))
        if params.ESTIMATE_DIMENSIONS:
            measure_z[0:3] = meas.z[0:3]
        else:
            measure_z[0:3] = meas.z
        x = np.zeros((6,1))
        if params.ESTIMATE_DIMENSIONS:
            x = np.zeros((9,1)) # three more state variables for width, length, height
            x[6:9] = 1 # Initial guess for width, length and height = 1m
        x[0:3] = (meas.sensor.sens_to_veh * measure_z)[0:3]
        self.x = x

        P = np.zeros((6,6))
        if params.ESTIMATE_DIMENSIONS:
            P = np.zeros((9,9))
            P[6:9,6:9] = np.matrix([
                [params.sigma_dims, 0, 0],
                [0, params.sigma_dims, 0],
                [0, 0, params.sigma_dims]
            ])
        P[0:3,0:3] = M_rot * meas.R[0:3,0:3] * M_rot.transpose()
        P[3:6,3:6] = np.matrix([
            [params.sigma_p44**2, 0, 0],
            [0, params.sigma_p55**2, 0],
            [0, 0, params.sigma_p66**2],
        ])
        if params.USE_BICYCLE_MODEL:
            P[3:6,3:6] = np.matrix([
            [0.5**2, 0, 0],
            [0, params.sigma_p55**2, 0],
            [0, 0, 0.5**2],
        ])
        self.P = P

        self.state = 'created' # Just 'created' to prevent deletion of track in the same frame it is created
        self.score = 0
        
        ############
        # END student code
        ############ 
               
        # other track attributes
        self.id = id
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        self.yaw =  np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
        self.t = meas.t

    def set_x(self, x):
        self.x = x
        
    def set_P(self, P):
        self.P = P  
        
    def set_t(self, t):
        self.t = t  
        
    def update_attributes(self, meas):
        # use exponential sliding average to estimate dimensions and orientation
        if meas.sensor.name == 'lidar':
            c = params.weight_dim
            self.width = c*meas.width + (1 - c)*self.width
            self.length = c*meas.length + (1 - c)*self.length
            self.height = c*meas.height + (1 - c)*self.height
            M_rot = meas.sensor.sens_to_veh
            self.yaw = np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
        
        
###################        

class Trackmanagement:
    '''Track manager with logic for initializing and deleting objects'''
    def __init__(self):
        self.N = 0 # current number of tracks
        self.track_list = []
        self.last_id = -1
        self.result_list = []
        
    def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list):  
        # decrease score for unassigned tracks
        for i in unassigned_tracks:
            track = self.track_list[i]
            # check visibility    
            if meas_list: # if not empty
                if meas_list[0].sensor.in_fov(track.x):
                    track.score = max(track.score - 1/params.window, 0)

        def check_covariance_is_too_big(track):
            return track.P[0, 0] > params.max_P or track.P[1, 1] > params.max_P

        # delete old tracks   
        for track in self.track_list:
            if check_covariance_is_too_big(track):
                self.delete_track(track)
            else:
                if track.state == 'confirmed':
                    if track.score <= params.delete_threshold:
                        self.delete_track(track)
                if track.state == 'initialized':
                    if track.score < 1/params.window:
                        self.delete_track(track)
                if track.state == 'tentative':
                    if track.score <= 0.0:
                        self.delete_track(track)

        ############
        # END student code
        ############ 
            
        # initialize new track with unassigned measurement
        for j in unassigned_meas: 
            if meas_list[j].sensor.name == 'lidar': # only initialize with lidar measurements
                self.init_track(meas_list[j])
            
    def addTrackToList(self, track):
        self.track_list.append(track)
        self.N += 1
        self.last_id = track.id

    def init_track(self, meas):
        track = Track(meas, self.last_id + 1)
        self.addTrackToList(track)

    def delete_track(self, track):
        print(f"deleting track no {track.id}")
        self.track_list.remove(track)
        
    def handle_updated_track(self, track):      
        track.state = 'initialized'
        track.score = min(track.score + 1/params.window, 1)
        if(track.id == '2'):
            print(f" Updating score of track {track.id} to {track.score}")
        if track.score > 0.2:
            track.state = 'tentative'
        if track.score >= params.confirmed_threshold:
            track.state = 'confirmed'
            #if(track.id != 1 and track.id != 0):
            #    print(track.id)
            #    exit(0)
        
        ############
        # END student code
        ############ 