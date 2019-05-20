"""
3D Pose and Path Estimation of the Planar Robot Using Extended Kalman Filter
Created on Aug 2016
Updated on May 2019
By Sina M.Baharlou (Sina.baharlou@gmail.com)
Web page: www.sinabaharlou.com
"""

# -- Import the required libraries --
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ekf
from RosDB import RosDB

# -- Constants and Definitions --
POSITION = 'position'
ORIENTATION = 'orientation'
LINEAR_A = 'linear_a'
ANGULAR_V = 'angular_v'
LINEAR_T = 'linear_t'
ANGULAR_T = 'angular_t'
POSE_COV = 'pose_cov'
TWIST_COV = 'twist_cov'
ORI_COV = 'ori_cov'
TIME = 'time'
ODOM = '/odom'
IMU = '/imu/data'


# -- 3d plot --
def plot3(a, b, c):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(a, b, c, label='Estimated path using EKF')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def main():
    # -- Open Ros-bag file --
    db = RosDB("ros.bag", True)
    db.load_bag()  # -- load contents --

    # -- Get sensor values --
    odom = db.get_odom_values()
    imu = db.get_imu_values()

    # -- Get & Interpolate the data --
    print("Interpolating imu date...")
    lv_x = odom[LINEAR_T][:, 0]
    w = odom[ANGULAR_T][:, 2]
    time_odo = odom['time']
    time_imu = imu['time']

    # -- Get odometry data --
    x_odom = odom[POSITION][:, 0]
    y_odom = odom[POSITION][:, 1]

    # -- Get IMU data --
    roll = imu[ORIENTATION][:, 0]
    pitch = imu[ORIENTATION][:, 1]
    yaw = imu[ORIENTATION][:, 2]

    # -- Interpolate the data --
    roll = np.interp(time_odo, time_imu, roll)
    pitch = np.interp(time_odo, time_imu, pitch)
    yaw = np.interp(time_odo, time_imu, yaw)

    # -- Define EKF Parameters --
    # H matrix
    h_mat = np.matrix(np.zeros([3, 6]))
    h_mat[0, 3] = 1
    h_mat[1, 4] = 1
    h_mat[2, 5] = 1

    # -- V matrix --
    v_mat = np.matrix(np.eye(3))

    # -- IMU measurement noise --
    m_noise = np.matrix(imu[ORI_COV])

    # -- Prior gaussian --
    prior = ekf.Gaussian([0, 0, 0, roll[0], pitch[0], yaw[0]], np.eye(6, 6) * 0.01)

    # -- Create filter --
    ekf_filter = ekf.ExtendedKalmanFilter(prior)

    # -- Define the state parameters
    state = prior
    size = len(time_odo)
    current_time = 0
    pos = np.zeros([size, 3])

    # -- Main loop for ekf path estimation
    print("Filtering (Path estimation using ekf) ...")
    for i in range(size):
        # -- Get Dt --
        dt = time_odo[i] - current_time
        current_time = time_odo[i]

        # -- Add current position to the list --
        pos[i] = [state.get_mean()[0], state.get_mean()[1], state.get_mean()[2]]

        # -- Create control gaussian --
        control = ekf.Gaussian([lv_x[i] * dt, w[i] * dt], np.matrix([[0.01, 0], [0, 0.01]]))

        # -- Perform the prediction --
        ekf_filter.predict(control)

        # -- Create measurement gaussian --
        measurement = ekf.Gaussian([roll[i], pitch[i], yaw[i]], m_noise)

        # -- Perform the update --
        state = ekf_filter.update(h_mat, v_mat, measurement)

    # -- Plot estimated 2d path --
    plt.plot(pos[:, 0], pos[:, 1], 'black')
    plt.plot(x_odom, y_odom, 'b')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Estimated path using ekf')
    plt.legend("ekf")
    plt.legend("odometry")
    plt.grid()
    plt.show()

    # -- Plot estimated 3d path --
    plot3(pos[:, 0], pos[:, 1], pos[:, 2])


if __name__ == "__main__":
    os.system('reset')
    main()
