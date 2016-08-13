# -- EKF by Sina moayed Baharlou (1672657)

import numpy as np
import transformations as tf
from math import *

class Gaussian:
	# -- constructor 
	def __init__(self,mean,covariance):
		self.__mean=np.array(mean);
		self.__covariance=np.matrix(covariance);
	
	# -- get expected value
	def get_mean(self):
		return self.__mean;

	# -- get covariance
	def get_covariance(self):
		return self.__covariance;


class ExtendedKalmanFilter:

	# -- constructor 
	def __init__(self,prior_state):	
		self.estimated_state=prior_state;

	# -- transition function (system model)
	def transition(self,state,control):	
		tf_prev=self.state_to_tfm(state)
		mat=np.matrix(tf.euler_matrix(0,0,control[1],'rxyz'));
		mat[0,3]= control[0];
		#mat[1,3]= control[1];
		#mat[2,3]= control[2];
		return self.tfm_to_state(tf_prev*mat);	
		

	# -- transition function (system model)
	def transition_(self,state,control):	
		
		v=control[0];
		w=control[1]; # dt included
		r,p,y=(state[3],state[4],state[5])
		
		states=[state[0]+v*cos(p)*cos(y+w/2),state[1]+v*cos(p)*sin(y+w/2),state[2]-v*sin(p),state[3],state[4],state[5]+w]

		
		return self.tfm_to_state(self.state_to_tfm(states));	



	# -- time update (predict) -- 
	def predict(self,control):	
		

		# -- get control mean & covariance 
		control_m=control.get_mean();		
		control_c=control.get_covariance();
		
		# -- get previous state mean & covariance (Xhat(k-1) , P(k-1)) 
		state_m=self.estimated_state.get_mean();
		state_c=self.estimated_state.get_covariance();

		# -- Calculate priori estimate of the next state (Xhat-(k)) 
		new_state_m=self.transition(state_m,control_m);

		# -- Calculate jacobian matrix of partial derivatives of f(transition function) w.r.t  X(state vector)
		j_t=self.t_jacobian(state_m,control_m)
		# -- Calculate jacobian matrix of partial derivatives of f(transition function) w.r.t  W(control vector)
		j_w=self.w_jacobian(state_m,control_m)

		# -- update the new covariance  (P-(k)=A*P*A' + W*Q*W')
		new_state_c=j_t* state_c *j_t.transpose() + j_w*control_c*j_w.transpose();

		# -- assign estimated state 
		self.estimated_state=Gaussian(new_state_m,new_state_c);

		return self.estimated_state;
		

	# -- measurement update 
	def update(self,h,v,measurement):
		
		
		# -- get measurement mean & covariance (Z & Rk)
		measure_m=np.matrix(measurement.get_mean()).transpose();
		measure_c=measurement.get_covariance();

		# -- get current estimated state mean & covariance (Xhat-(k) , P-(k)) 
		state_m=np.matrix(self.estimated_state.get_mean()).transpose();
		state_c=self.estimated_state.get_covariance();
		size=len(state_m)

		# -- calculate innovation 
		innovation=measure_m-h*state_m;

		# -- calculate innovation covariance
		innovation_cov=h*state_c*h.transpose() + v*measure_c*v.transpose();

		# -- calculate kalman gain
		kalman_gain=state_c*h.transpose() * np.linalg.pinv(innovation_cov);
		
		# -- calculate posteriori estimate --
		new_state_m=state_m+ kalman_gain*innovation;
		new_state_c=(np.eye(size)-kalman_gain*h)* state_c;


		temp_state=[new_state_m[0,0],new_state_m[1,0],new_state_m[2,0],
			    new_state_m[3,0],new_state_m[4,0],new_state_m[5,0]]
		# -- assign estimated state 
		self.estimated_state=Gaussian(temp_state,new_state_c);
		
		return self.estimated_state;
		


	#-- jacobian matrix of partial derivatives of F w.r.t  X
	def t_jacobian(self,state,control):
		j=np.matrix(np.eye(6,6));
		v=control[0];
		r,p,y=(state[3],state[4],state[5])

		j[0,4]= -v*cos(y)*sin(p);
		j[0,5]=  -v*cos(p)*sin(y);

		j[1,4]= -v*sin(p)*sin(y);
		j[1,5]=  v*cos(p)*cos(y);

		j[2,4]=  -v*cos(p);

		return j;


	#-- jacobian matrix of partial derivatives of F w.r.t  W
	def w_jacobian(self,state,control):
		j=np.matrix(np.zeros([6,2]));
		v=control[0];
		r,p,y=(state[3],state[4],state[5])

		j[0,0]= cos(p)*cos(y);
		j[1,0]=  cos(p)*sin(y);
		j[2,0]=  -sin(p);
		j[5,1]=  1;# or dt?
		
		return j;


	# -- Convert state vector to transformation matrix 
	def state_to_tfm(self,state):
		mat=np.matrix(tf.euler_matrix(state[3],state[4],state[5]));
		mat[0,3]= state[0]
		mat[1,3]= state[1]
		mat[2,3]= state[2]
		return mat

	# -- Convert  transformation matrix to state vector 
	def tfm_to_state(self,tfm):
		euler=tf.euler_from_matrix(tfm)
		return [tfm[0,3],tfm[1,3],tfm[2,3], euler[0],euler[1],euler[2]];
				
					
				
