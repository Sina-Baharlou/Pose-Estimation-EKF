"""
3D Pose and Path Estimation of the Planar Robot Using Extended Kalman Filter
Created on Aug 2016
Updated on May 2019
By Sina M.Baharlou (Sina.baharlou@gmail.com)
Web page: www.sinabaharlou.com
"""

# -- Import the required libraries --
import numpy as np
import transformations as tf
from math import *


# -- Gaussian Distribution Class --
class Gaussian:
    # -- Constructor
    def __init__(self, mean, covariance):
        self.__mean = np.array(mean)
        self.__covariance = np.matrix(covariance)

    # -- get expected value
    def get_mean(self):
        return self.__mean

    # -- get covariance
    def get_covariance(self):
        return self.__covariance


# -- Extended Kamlan Filter Class --
class ExtendedKalmanFilter:

    # -- Constructor --
    def __init__(self, prior_state):
        self.estimated_state = prior_state

    # -- Transition function (system model) --
    def transition(self, state, control):
        tf_prev = self.state_to_tfm(state)
        mat = np.matrix(tf.euler_matrix(0, 0, control[1], 'rxyz'))
        mat[0, 3] = control[0]
        # mat[1,3]= control[1];
        # mat[2,3]= control[2];
        return self.tfm_to_state(tf_prev * mat)

    # -- Transition function (system model) --
    def transition_(self, state, control):
        v = control[0]
        w = control[1]  # dt included
        r, p, y = (state[3], state[4], state[5])

        states = [state[0] + v * cos(p) * cos(y + w / 2), state[1] + v * cos(p) * sin(y + w / 2), state[2] - v * sin(p),
                  state[3], state[4], state[5] + w]

        return self.tfm_to_state(self.state_to_tfm(states))

    # -- Time update (predict) --
    def predict(self, control):
        # -- Get control mean & covariance --
        control_m = control.get_mean()
        control_c = control.get_covariance()

        # -- Get previous state mean & covariance (Xhat(k-1) , P(k-1)) --
        state_m = self.estimated_state.get_mean()
        state_c = self.estimated_state.get_covariance()

        # -- Calculate priori estimate of the next state (Xhat-(k)) --
        new_state_m = self.transition(state_m, control_m)

        # -- Calculate jacobian matrix of partial derivatives of f(transition function) w.r.t  X(state vector) --
        j_t = self.t_jacobian(state_m, control_m)
        # -- Calculate jacobian matrix of partial derivatives of f(transition function) w.r.t  W(control vector) --
        j_w = self.w_jacobian(state_m, control_m)

        # -- Update the new covariance  (P-(k)=A*P*A' + W*Q*W') --
        new_state_c = j_t * state_c * j_t.transpose() + j_w * control_c * j_w.transpose()

        # -- Assign estimated state --
        self.estimated_state = Gaussian(new_state_m, new_state_c)

        return self.estimated_state

    # -- Measurement update --
    def update(self, h, v, measurement):
        # -- Get measurement mean & covariance (Z & Rk) --
        measure_m = np.matrix(measurement.get_mean()).transpose()
        measure_c = measurement.get_covariance()

        # -- Get current estimated state mean & covariance (Xhat-(k) , P-(k)) --
        state_m = np.matrix(self.estimated_state.get_mean()).transpose()
        state_c = self.estimated_state.get_covariance()
        size = len(state_m)

        # -- Calculate innovation --
        innovation = measure_m - h * state_m

        # -- Calculate innovation covariance --
        innovation_cov = h * state_c * h.transpose() + v * measure_c * v.transpose()

        # -- Calculate kalman gain --
        kalman_gain = state_c * h.transpose() * np.linalg.pinv(innovation_cov)

        # -- Calculate posterior estimate --
        new_state_m = state_m + kalman_gain * innovation
        new_state_c = (np.eye(size) - kalman_gain * h) * state_c

        temp_state = [new_state_m[0, 0], new_state_m[1, 0], new_state_m[2, 0],
                      new_state_m[3, 0], new_state_m[4, 0], new_state_m[5, 0]]
        # -- Assign estimated state --
        self.estimated_state = Gaussian(temp_state, new_state_c)

        return self.estimated_state

    # -- Jacobian matrix of partial derivatives of F w.r.t X --
    @staticmethod
    def t_jacobian(state, control):
        j = np.matrix(np.eye(6, 6))
        v = control[0]
        r, p, y = (state[3], state[4], state[5])

        # -- Determine the Jacobian --
        j[0, 4] = -v * cos(y) * sin(p)
        j[0, 5] = -v * cos(p) * sin(y)
        j[1, 4] = -v * sin(p) * sin(y)
        j[1, 5] = v * cos(p) * cos(y)
        j[2, 4] = -v * cos(p)

        return j

    # -- Jacobian matrix of partial derivatives of F w.r.t W --
    @staticmethod
    def w_jacobian(state, control):
        j = np.matrix(np.zeros([6, 2]))
        v = control[0] # Not used in this problem
        r, p, y = (state[3], state[4], state[5])

        # -- Determine the Jacobian --
        j[0, 0] = cos(p) * cos(y)
        j[1, 0] = cos(p) * sin(y)
        j[2, 0] = -sin(p)
        j[5, 1] = 1

        return j

    # -- Convert state vector to transformation matrix --
    @staticmethod
    def state_to_tfm(state):
        mat = np.matrix(tf.euler_matrix(state[3], state[4], state[5]))
        mat[0, 3] = state[0]
        mat[1, 3] = state[1]
        mat[2, 3] = state[2]
        return mat

    # -- Convert transformation matrix to state vector --
    @staticmethod
    def tfm_to_state(tfm):
        euler = tf.euler_from_matrix(tfm)
        return [tfm[0, 3], tfm[1, 3], tfm[2, 3], euler[0], euler[1], euler[2]]
