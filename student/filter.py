# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 
# params.dt

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        if params.ESTIMATE_DIMENSIONS:
            return np.matrix([
                [ 1,    0,      0,      params.dt,  0,          0,          0, 0, 0 ],
                [ 0,    1,      0,      0,          params.dt,  0,          0, 0, 0 ],
                [ 0,    0,      1,      0,          0,          params.dt,  0, 0, 0 ],
                [ 0,    0,      0,      1,          0,          0,          0, 0, 0 ],
                [ 0,    0,      0,      0,          1,          0,          0, 0, 0 ],
                [ 0,    0,      0,      0,          0,          1,          0, 0, 0 ],
                [ 0,    0,      0,      0,          0,          0,          1, 0, 0 ],
                [ 0,    0,      0,      0,          0,          0,          0, 1, 0 ],
                [ 0,    0,      0,      0,          0,          0,          0, 0, 1 ],
            ])
        return np.matrix([
            [ 1,    0,      0,      params.dt,  0,          0           ],
            [ 0,    1,      0,      0,          params.dt,  0           ],
            [ 0,    0,      1,      0,          0,          params.dt   ],
            [ 0,    0,      0,      1,          0,          0           ],
            [ 0,    0,      0,      0,          1,          0           ],
            [ 0,    0,      0,      0,          0,          1           ],
        ])
        
        ############
        # END student code
        ############ 

    def bicycle_non_linear_F(self, state_vector):
        '''Non-linear bicycle model with vector state [x, y, z, theta, v, delta]
            Where x, y, z is the position in 3D
            x and y are updated from bicycle velocity v, body angle theta, and front wheel angle delta
            z is assumed to be preserved throughout the trajectory in this simple model
        '''
        x = state_vector[0, 0]
        y = state_vector[1, 0]
        z = state_vector[2, 0]
        theta = state_vector[3, 0]
        v = state_vector[4, 0]
        delta = state_vector[5, 0]
        return np.matrix([
            [ x + v*np.cos(theta) * params.dt ],  # x 
            [ y + v*np.sin(theta) * params.dt ],  # y
            [ z ],                                # z
            [ v*np.sin(delta) ],                  # theta
            [ v ],                                # v
            [ delta ]                             # delta
        ])

    def bicycle_linear_F(self, x):
        '''Non-linear bicycle model linearized around point [x_0, y_0, z_0, theta_0, v_0, delta_0]'''
        theta = x[3, 0]
        v = x[4, 0]
        delta = x[5, 0]
        return np.matrix([
            [ 1,  0,  0,  - v * np.sin(theta) * params.dt,  np.cos(theta) * params.dt,    0 ],
            [ 0,  1,  0,  v * np.cos(theta) * params.dt,    np.sin(theta) * params.dt,    0 ],
            [ 0,  0,  1,  0,                                0,                            0 ],
            [ 0,  0,  0,  0,                                np.sin(delta),                v*np.cos(delta) ],
            [ 0,  0,  0,  0,                                1,                            0 ],
            [ 0,  0,  0,  0,                                0,                            1 ]
        ])

    def Q(self):
        if params.ESTIMATE_DIMENSIONS:
            return np.matrix([
            [ 1/3 * params.q**2 * params.dt**3, 0, 0, 1/2 * params.q**2 * params.dt**2, 0, 0, 0, 0, 0 ],
            [ 0, 1/3 * params.q**2 * params.dt**3, 0, 0, 1/2 * params.q**2 * params.dt**2, 0, 0, 0, 0 ],
            [ 0, 0, 1/3 * params.q**2 * params.dt**3, 0, 0, 1/2 * params.q**2 * params.dt**2, 0, 0, 0 ],
            [ 1/2 * params.q**2 * params.dt**2, 0, 0, params.q**2 * params.dt, 0, 0, 0, 0, 0 ],
            [ 0, 1/2 * params.q**2 * params.dt**2, 0, 0, params.q**2 * params.dt, 0, 0, 0, 0 ],
            [ 0, 0, 1/2 * params.q**2 * params.dt**2, 0, 0, params.q**2 * params.dt, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
            ])
        return np.matrix([
            [ 1/3 * params.q**2 * params.dt**3, 0, 0, 1/2 * params.q**2 * params.dt**2, 0, 0 ],
            [ 0, 1/3 * params.q**2 * params.dt**3, 0, 0, 1/2 * params.q**2 * params.dt**2, 0 ],
            [ 0, 0, 1/3 * params.q**2 * params.dt**3, 0, 0, 1/2 * params.q**2 * params.dt**2 ],
            [ 1/2 * params.q**2 * params.dt**2, 0, 0, params.q**2 * params.dt, 0, 0 ],
            [ 0, 1/2 * params.q**2 * params.dt**2, 0, 0, params.q**2 * params.dt, 0 ],
            [ 0, 0, 1/2 * params.q**2 * params.dt**2, 0, 0, params.q**2 * params.dt ],
        ])
    
    def bicycle_Q(self, state_vector):
        '''Linearized and integrated process noise Q around point [x_0, y_0, z_0, theta_0, v_0, delta_0]'''
        self.q_v = 50
        self.q_d = 1
        # x: state_vector[0], y: state_vector[1], z: state_vector[2], theta: state_vector[3], v: state_vector[4], delta: state_vector[5]
        theta = state_vector[3, 0]
        v = state_vector[4, 0]
        delta = state_vector[5, 0]
        return np.matrix([
            [ 1/3 * v**2 * self.q_v * np.sin(theta)**2 * params.dt**3,                 -1/3 * v**2 * self.q_v * np.sin(theta) * np.cos(theta) * params.dt**3,    0,     -1/2 * v * self.q_v * np.sin(delta) * np.sin(theta) * params.dt**2,                 -1/2 * v * self.q_v * np.sin(theta) * params.dt**2,     0 ],
            [ -1/3 * v**2 * self.q_v * np.sin(theta) * np.cos(theta) * params.dt**3,    1/3 * v**2 * self.q_v * np.cos(theta)**2 * params.dt**3,                 0,     1/2 * v * self.q_v * np.sin(delta) * np.cos(theta) * params.dt**2,                  1/2 * v * self.q_v * np.cos(theta) * params.dt**2,      0 ],
            [ 0,                                                                        0,                                                                       0,     0,                                                                                  0,                                                      0 ],
            [ -1/2 * v * self.q_v * np.sin(delta) * np.sin(theta) * params.dt**2,       1/2 * v * self.q_v * np.sin(delta) * np.cos(theta) * params.dt**2,       0,     (v**2 * self.q_d * np.cos(delta)**2 + self.q_v * np.sin(delta)**2) * params.dt,     self.q_v * np.sin(delta) * params.dt,                   v * self.q_d * np.cos(delta) ],
            [ -1/2 * v * self.q_v * np.sin(theta) * params.dt**2,                       1/2 * v * self.q_v * np.cos(theta) * params.dt**2,                       0,     self.q_v * np.sin(delta) * params.dt,                                               self.q_v * params.dt,                                   0 ],
            [ 0,                                                                        0,                                                                       0,     v * self.q_d * np.cos(delta) * params.dt,                                           0,                                                      self.q_d * params.dt ]
        ])
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        if params.USE_BICYCLE_MODEL:
            new_x = self.bicycle_non_linear_F(track.x)
            new_P = self.bicycle_linear_F(track.x) * track.P * self.bicycle_linear_F(track.x).transpose() + self.bicycle_Q(track.x)
        else:
            new_x = self.F() * track.x
            new_P = self.F() * track.P * self.F().transpose() + self.Q()
        track.set_x(new_x)
        track.set_P(new_P)
        
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        y = self.gamma(track, meas)
        S = self.S(track, meas, meas.sensor.get_H(track.x))
        K = track.P * meas.sensor.get_H(track.x).transpose() * np.linalg.inv(S)
        I = np.eye(track.x.shape[0])
        new_x = track.x + (K * y)
        new_P = (I - K * meas.sensor.get_H(track.x)) * track.P
        track.set_x(new_x)
        track.set_P(new_P)

        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        return meas.z - meas.sensor.get_hx(track.x)
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        return H * track.P * H.transpose() + meas.R
        
        ############
        # END student code
        ############ 