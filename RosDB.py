"""
3D Pose and Path Estimation of the Planar Robot Using Extended Kalman Filter
Created on Aug 2016
Updated on May 2019
By Sina M.Baharlou (Sina.baharlou@gmail.com)
Web page: www.sinabaharlou.com
"""

# -- Import the required libraries --
import rosbag
from sys import stdout
import numpy as np
import transformations as tf

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

ori_cov_size = [3, 3]
pose_cov_size = [6, 6]
twist_cov_size = [6, 6]


# -- Ros Database class --
class RosDB:

    # -- Constructor --
    def __init__(self, rosbag_path, enable_debug):
        self.rosbag_path = rosbag_path  # -- set rosbag path
        self.enable_debug = enable_debug  # -- enable debugging ( print log messages)
        self.bag = None  # -- rosbag class
        self.bag_contents = None  # -- bag contents
        self.bag_info = None  # -- bag info
        self.bag_topics = None  # -- bag topics

    # -- Load bag file (should be called at the beginning) --
    def load_bag(self):
        # -- Load ros bag file --
        self.debug("Open rosbag file : '%s' ... " % self.rosbag_path)
        self.bag = rosbag.Bag(self.rosbag_path)

        # -- Get contents --
        self.debug("Get bag contents...")
        self.bag_contents = self.bag.read_messages()

        # -- Get info --
        self.bag_info = self.bag.get_type_and_topic_info()

    # -- Get rosbag topics --
    def get_topics(self):
        if self.bag_info is not None:
            return self.bag_info.topics.keys()
        else:
            self.debug("run 'load_bag' first.")
            return None

    # -- Print rosbag topics --
    def print_topics(self):
        if self.bag_info is None:
            self.debug("run 'load_bag' first.")
            return None

        # -- Get the topics --
        topics = self.get_topics()

        # -- Start printing the details --
        print("Available topics are :")
        for topic in topics:
            print("\t", topic)
            topic_info = self.bag_info[1][topic]
            print("\tMessage type :%s" % topic_info.msg_type)
            print("\tMessage count :%d" % topic_info.message_count)
            print("\tFrequency :%d" % topic_info.frequency)
            print("\n")

    # -- Get start,end,duration time --
    def get_time(self):
        if self.bag is None:
            self.debug("run 'load_bag' first.")
            return None

        duration = self.bag.get_end_time() - self.bag.get_start_time()
        return self.bag.get_start_time(), self.bag.get_end_time(), duration

    # -- Get Odometry values in a numpy matrix format --
    def get_odom_values(self):
        if self.bag_info is None:
            self.debug("run 'load_bag' first.")
            return None

        # -- Get message count --
        message_count = self.bag_info[1][ODOM].message_count

        # -- Initialize the matrix
        odometry = {
            POSITION: np.zeros([message_count, 3]),
            ORIENTATION: np.zeros([message_count, 3]),  # It's quaternion
            LINEAR_T: np.zeros([message_count, 3]),
            ANGULAR_T: np.zeros([message_count, 3]),  # It's quaternion
            TIME: np.zeros([message_count]),
            TWIST_COV: [],
            POSE_COV: []}

        index = 0
        start_time = None
        portion = (message_count / 10)
        self.debug("Fetching odometry data...")

        # -- Get the data --
        for subtopic, msg, t in self.bag.read_messages(ODOM):

            # -- Shortcuts --
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation

            lt = msg.twist.twist.linear
            at = msg.twist.twist.angular

            # -- Convert quaternion to euler angles --
            quaternion = (ori.x, ori.y, ori.z, ori.w)
            euler = tf.euler_from_quaternion(quaternion)

            # -- Set initial T and get covariance
            if start_time is None:
                start_time = t
                odometry[POSE_COV] = np.matrix(msg.pose.covariance).reshape(pose_cov_size)
                odometry[TWIST_COV] = np.matrix(msg.twist.covariance).reshape(twist_cov_size)

            # -- Add values --
            odometry[POSITION][index] = [pos.x, pos.y, pos.z]
            odometry[ORIENTATION][index] = [euler[0], euler[1], euler[2]]
            odometry[LINEAR_T][index] = [lt.x, lt.y, lt.z]
            odometry[ANGULAR_T][index] = [at.x, at.y, at.z]
            odometry[TIME][index] = t.to_sec() - start_time.to_sec()

            # -- Show progress
            if self.enable_debug and index % portion == 0:
                stdout.write(".")
                stdout.flush()

            index += 1

        return odometry

    # -- get IMU values in a numpy matrix format  --
    def get_imu_values(self):
        if self.bag_info is None:
            self.debug("run 'load_bag' first.")
            return None

        # -- Get message count --
        message_count = self.bag_info[1][IMU].message_count

        # -- Initialize the matrix --
        imu = {ORIENTATION: np.zeros([message_count, 3]),
               # LINEAR_A:np.zeros([message_count,3]),
               # ANGULAR_V:np.zeros([message_count,3]),
               TIME: np.zeros([message_count]),
               ORI_COV: np.zeros([3, 3])}

        index = 0
        start_time = None
        portion = (message_count / 10)
        self.debug("Fetching imu data ")

        # -- Get the data --
        for subtopic, msg, t in self.bag.read_messages(IMU):

            # -- Shortcuts --
            ori = msg.orientation
            # la=msg.linear_acceleration;
            # av=msg.angular_velocity;

            # -- Convert quaternion to euler angles --
            quaternion = (ori.x, ori.y, ori.z, ori.w)
            euler = tf.euler_from_quaternion(quaternion)

            # -- Set initial T and get covariance --
            if start_time is None:
                start_time = t
                imu[ORI_COV] = np.matrix(msg.orientation_covariance).reshape(ori_cov_size)

            # -- Add values
            imu[ORIENTATION][index] = [euler[0], euler[1], euler[2]]
            # imu[LINEAR_A][index]=[la.x,la.y,la.z];
            # imu[ANGULAR_V][index]=[av.x,av.y,av.z];
            imu[TIME][index] = t.to_sec() - start_time.to_sec()

            if self.enable_debug and index % portion == 0:
                stdout.write(".")
                stdout.flush()
            index += 1

        return imu

    def debug(self, message):
        if not self.enable_debug:
            return
        print("RosDB: %s" % message)
