# --3d Pose estimation of a 2d planar robot using EKF by  Sina moayed Baharlou (1672657)

import rosbag
import roslib
import os
import numpy as np
import transformations as tf
from math import *
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
from RosDB import RosDB
import ekf 

# -- Constants and Definitions -- 

POSITION='position'
ORIENTATION='orientation'
LINEAR_A='linear_a';
ANGULAR_V='angular_v';
LINEAR_T='linear_t';
ANGULAR_T='angular_t';
POSE_COV='pose_cov';
TWIST_COV='twist_cov';
ORI_COV='ori_cov';
TIME='time'
ODOM='/odom'
IMU='/imu/data'



# -- 3d plot -- 
def plot3(a,b,c,mark="o",col="r"):
	
	fig = pylab.figure()
	ax = fig.gca(projection='3d')
	ax.plot(a, b, c, label='Estimated path using EKF')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.legend()
	pylab.show()



# -- Main Procedure ------

# clear terminal 
os.system("clear");

# -- open rosbag 
db=RosDB("ros.bag",True)
db.load_bag() # -- load contents 

# -- get sensor data
odom=db.get_odom_values()
imu=db.get_imu_values()




# -- Get & Interpolate the data -- 

print "Interpolating imu date..."
lv_x=odom[LINEAR_T][:,0];
w=odom[ANGULAR_T][:,2];
time_odo=odom['time']
time_imu=imu['time']

x_odom=odom[POSITION][:,0]
y_odom=odom[POSITION][:,1]


roll=imu[ORIENTATION][:,0];
pitch=imu[ORIENTATION][:,1];
yaw=imu[ORIENTATION][:,2];


roll=np.interp(time_odo,time_imu,roll);
pitch=np.interp(time_odo,time_imu,pitch);
yaw=np.interp(time_odo,time_imu,yaw);



# -- EKF Parameters -- 
# H matrix 
H=np.matrix(np.zeros([3,6]))
H[0,3]=1
H[1,4]=1
H[2,5]=1

# V matrix 
V=np.matrix(np.eye(3))


# IMU measurement noise 
m_noise=np.matrix(imu[ORI_COV]);

# Prior gaussian 
prior=ekf.Gaussian([0,0,0,roll[0],pitch[0],yaw[0]],np.eye(6,6)*0.01)

# Create filter 
filter=ekf.ExtendedKalmanFilter(prior)


# -- 
state=prior;
size=len(time_odo);
current_time=0;
pos=np.zeros([size,3]);
#error=np.zeros(size);


print "Filtering (Path estimation using ekf) ..."
# -- Main loop for ekf path estimation 
for i in range( size):
	
	# -- get dt -- 
	dt=time_odo[i]-current_time;
	current_time=time_odo[i];

	# -- add current position to the list 
	pos[i]=[state.get_mean()[0],state.get_mean()[1],state.get_mean()[2]];
	#error[i]=np.linalg.det(state.get_covariance());

	
	# -- create control gaussian 
	control=ekf.Gaussian([lv_x[i]*dt,w[i]*dt],np.matrix([ [0.01,0],[0,0.01]]))

	# -- do the prediction  
	filter.predict(control)

	# -- create measurement gaussian 
	measurement=ekf.Gaussian([roll[i],pitch[i],yaw[i]],m_noise);
	
	# -- do the update 
	state=filter.update(H,V,measurement);
	


# plot estimated 2d path
pylab.plot(pos[:,0],pos[:,1],'black');
pylab.plot(x_odom,y_odom,'b');

pylab.xlabel('X')
pylab.ylabel('Y')
pylab.title('Estimated path using ekf')
pylab.legend("ekf")
pylab.legend("odometry")
pylab.grid();
pylab.show();

# plot estimated 3d path
plot3(pos[:,0],pos[:,1],pos[:,2])


