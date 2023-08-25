# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params

from munkres import Munkres # Add 'munkres' to requirements.txt

m = Munkres()

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        if params.GLOBAL_NEAREST_NEIGHBOR:
            self.associations = None
            self.last_association_index = -1
        
    def associate(self, track_list, meas_list, KF, frame=-1):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = [] # reset lists
        self.unassigned_meas = []
                    
        if len(meas_list) > 0:
            self.unassigned_meas = list(range(len(meas_list)))
        if len(track_list) > 0:
            self.unassigned_tracks = list(range(len(track_list)))
        if len(meas_list) > 0 and len(track_list) > 0: 
            self.association_matrix = np.matrix(np.ones((len(track_list), len(meas_list)))*np.inf) # reset matrix
            for i in range(len(track_list)):
                for j in range(len(meas_list)):
                    self.association_matrix[i,j] = self.MHD(track_list[i], meas_list[j], KF)
        
        if params.GLOBAL_NEAREST_NEIGHBOR:
            # Using the Hungarian Algorithm to compute the best associations globally
            self.associations = m.compute(self.association_matrix.A.tolist())
            self.last_association_index = 0

        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):

        if params.GLOBAL_NEAREST_NEIGHBOR:
            if self.last_association_index == len(self.associations):
                return np.nan, np.nan
            update_track = self.associations[self.last_association_index][0]
            update_meas = self.associations[self.last_association_index][1]
            self.last_association_index += 1
            self.unassigned_tracks.remove(update_track) 
            self.unassigned_meas.remove(update_meas)
            return update_track, update_meas

        closest_index = np.argmin(self.association_matrix)
        closest_ij = np.unravel_index(closest_index, self.association_matrix.shape) 

        if self.association_matrix[closest_ij] == np.inf:
            return np.nan, np.nan

        update_track = self.unassigned_tracks[closest_ij[0]]
        update_meas = self.unassigned_meas[closest_ij[1]]

        # remove from list
        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)
        self.association_matrix = np.delete(np.delete(self.association_matrix, closest_ij[0], axis=0), closest_ij[1], axis=1)
            
        ############
        # END student code
        ############ 
        return update_track, update_meas

    def gating(self, MHD, sensor): 
        nd = sensor.dim_meas
        return chi2.ppf(MHD, nd) <= params.gating_threshold
        
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        return KF.gamma(track, meas).transpose() * np.linalg.inv(KF.S(track, meas, meas.sensor.get_H(track.x))) * KF.gamma(track, meas)
        
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF, frame=-1):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF, frame)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])

            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)