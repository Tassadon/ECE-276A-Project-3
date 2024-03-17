import numpy as np
from pr3_utils import *
from scipy.linalg import expm
import tqdm
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':

	# Load the measurements
	dataset = "10"
	filename = f"../data/{dataset}.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
	
	#cap = cv2.VideoCapture(f"../data/{dataset}_video_every10frames.avi")
	#length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

	#ret, frame = cap.read()
	#plt.imshow(frame)
	#print(frame.shape)

	zeta = np.concatenate([linear_velocity, angular_velocity])
	zeta_hat = axangle2twist(zeta.T)
	zeta_pointy_hat = axangle2adtwist(zeta.T)
	tau = np.diff(t)
	fsu = K[0,0]
	cu = K[0,2]
	fsv = K[1,1]
	cv = K[1,2]
	cam_T_imu = np.linalg.inv(imu_T_cam)

	K_s = np.array([[fsu, 0, cu, 0],
				 	[0, fsv, cv, 0],
					[fsu, 0, cu, -fsu*b],
					[0, fsv, cv, 0]])
	inv_K = np.linalg.inv(K)

	V = .01 * np.eye(4)

	print("pointy_hat_shape", zeta_pointy_hat[0].shape)
	print("features:", features.shape)
	print("K:", K.shape)
	print("b:", b)
	print(imu_T_cam.shape)
	print("linear_velocity:",linear_velocity.shape)
	print("angular velocity:", angular_velocity.shape)
	print("time:",t.shape)


	# (a) IMU Localization via EKF Prediction: no update
	T_0 = np.array([
					[1,0,0,0],
					[0,-1,0,0],
					[0,0,-1,0],
					[0,0,0,1]])
	#T_0 = np.eye(4)
	poses = [T_0]
	for i,tau_i in enumerate(tqdm.tqdm(np.squeeze(tau))):
		poses.append( poses[-1] @ expm(tau_i * zeta_hat[i]) )

	poses = np.array(poses)
	#visualize_trajectory_2d(np.transpose(poses,axes=(1,2,0)),path_name=f"IMU localization only for dataset {dataset}",show_ori=True)






	# (b) Landmark Mapping via EKF Update: no prediction
	mew_odometry = [T_0]
	W = np.diag([.01,.01,.01,.005,.005,.005])
	
	cov_odometry = [W]

	print(features[:,:,0].shape)
	c = np.apply_along_axis(lambda x: np.all(x != np.array([-1,-1,-1,-1])),0,features[:,:,0])

	print(features[:,c,0].shape)
	null_condition = np.array([-1,-1,-1,-1])
	landmark_means = {}
	landmark_cov = {}
	P = np.array([[1,0,0],
			   	  [0,1,0],
				  [0,0,1],
				  [0,0,0]])
	landmark_cov_init = 10^-3 * np.eye(4)

	for i,tau_i in enumerate(tqdm.tqdm(np.squeeze(tau))):
		mew_odometry.append( mew_odometry[-1] @ expm(tau_i * zeta_hat[i]) )
		cov_odometry.append(expm(-tau_i*zeta_pointy_hat[i]) @ cov_odometry[-1] @ expm(-tau_i*zeta_pointy_hat[i]).T \
					  + cov_odometry[0])
		
		available_feats_indices = np.nonzero(
			np.apply_along_axis(lambda x: np.all(x != null_condition),0,features[:,:,i])
			)[0]

		cam_T_world = cam_T_imu @ mew_odometry[-1] #mew_odometry[-1] is imu_T_world
		world_T_cam = np.linalg.inv(cam_T_world)

		for j in available_feats_indices:
			
			if j not in landmark_means.keys():
				d = features[0,j,i] - features[2,j,i]
				Z_0 = (K[0,0]*b)/d

				camera_coords = np.hstack((Z_0 * inv_K @ np.hstack( (  features[:2,j,i] , 1)) , 1) ) 

				landmark_means[j] = world_T_cam @ camera_coords
				
			else:
				print("ZA MEAN",landmark_means[j].shape)
				curr_landmark =  cam_T_world @ landmark_means[j]
				z_tilda = K_s @ projection(curr_landmark)

				print((mew_odometry[-1] @ landmark_means[j]).shape)

				H = -K_s @ projectionJacobian(curr_landmark) @ cam_T_imu @ P

				print(H.shape)
				
				#K_t = cov_odometry[-1] @ H.T @ np.linalg.inv(H@cov_odometry[-1]@H.T  + V)
				break
				#landmark_means[j] = mew_odometry @ expm(K_t( - z_tilda))
		
		

	#plt.scatter(x,y)
	#visualize_trajectory_2d(np.transpose(mew_odometry,axes=(1,2,0)),path_name=f"IMU localization only for dataset {dataset}",show_ori=True)




	# (c) Visual-Inertial SLAM

	# You may use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)
