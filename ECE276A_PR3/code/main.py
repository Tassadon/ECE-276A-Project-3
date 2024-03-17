import numpy as np
from pr3_utils import *
from scipy.linalg import expm
import tqdm
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':

	# Load the measurements
	dataset = "03"
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
	feats = features[:,::100,:]
	print("feats:", feats.shape)

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
		#The covariance doesnt affect mean here

	poses = np.array(poses)
	#visualize_trajectory_2d(np.transpose(poses,axes=(1,2,0)),path_name=f"IMU prediction only for dataset {dataset}",show_ori=True)

	# (b) Landmark Mapping via EKF Update: no prediction

	mew_odometry = [T_0]
	W = np.diag([.01,.01,.01,.005,.005,.005])
	cov_odometry = [W]
	null_condition = np.array([-1,-1,-1,-1])

	P = np.array([[1,0,0],
			   	  [0,1,0],
				  [0,0,1],
				  [0,0,0]])
	
	landmark_noise = .001 * np.eye(3)
	landmark_mean = np.zeros((3*feats.shape[1],1))
	landmark_seen = np.zeros((feats.shape[1],1))
	landmark_vanished = np.zeros((feats.shape[1],1))
	landmark_cov = {}
	count = 0
	IxV = .01 * np.eye(4)
	H_t = np.zeros((4*feats.shape[1],3*feats.shape[1]))

	for i,tau_i in enumerate(tqdm.tqdm(np.squeeze(tau))):
		
		mew_odometry.append( mew_odometry[-1] @ expm(tau_i * zeta_hat[i]) )
		cov_odometry.append(expm(-tau_i*zeta_pointy_hat[i]) @ cov_odometry[-1] @ expm(-tau_i*zeta_pointy_hat[i]).T \
					  + cov_odometry[0])
		
		available_feats_indices = np.nonzero(
			np.apply_along_axis(lambda x: np.all(x != null_condition),0,feats[:,:,i])
			)[0]
		
		cam_T_world = cam_T_imu @ mew_odometry[-1] #mew_odometry[-1] is imu_T_world
		world_T_cam = mew_odometry[-1] @ imu_T_cam
		
		for j in available_feats_indices:
			count += 1
			H_t = np.zeros((4*feats.shape[1],3*feats.shape[1]))

			if landmark_seen[j] == 0:

				landmark_seen[j] = 1
				d = feats[0,j,i] - feats[2,j,i]

				Z_0 = (K[0,0] * b) / d

				camera_coords = np.hstack((Z_0 * np.linalg.inv(K) @ np.hstack((feats[:2,j,i], 1)), 1))
				
				landmark_local = world_T_cam @ np.array([camera_coords]).T

				landmark_mean[(3*j):(3*j+3)] = landmark_local[:3]
				
			else:
				continue
				landmark_local = cam_T_world @ np.vstack([landmark_mean[(3*j):(3*j+3)],1])
				
				z_tilda = K_s @ projection(landmark_local.T).T

				H = (K_s @ projectionJacobian(landmark_local.T) @ cam_T_world @ P)[0] #4x3
				


				K_t = landmark_cov.get(j,landmark_noise) @ H.T @ \
					(H @ landmark_cov.get(j,landmark_noise) @ H.T + IxV)

				landmark_mean[(3*j):(3*j+3)] = np.array([np.squeeze(landmark_mean[(3*j):(3*j+3)]) + \
					K_t @(feats[:,j,i] - np.squeeze(z_tilda))]).T
				
				landmark_cov[j] = (np.eye(3) - K_t@H)@landmark_cov.get(j,landmark_noise)

	
	visualize_trajectory_2d_scatter(np.transpose(mew_odometry,axes=(1,2,0)),
						 landmark_mean[0::3],landmark_mean[1::3],
						 path_name=f"Visual Mapping only for dataset {dataset}",show_ori=True,dataset=dataset
						 ,fname="first_obs_only")
	'''
	mew_odometry = [T_0]
	W = np.diag([.01,.01,.01,.005,.005,.005])
	cov_odometry = [W]
	null_condition = np.array([-1,-1,-1,-1])

	P = np.array([[1,0,0],
			   	  [0,1,0],
				  [0,0,1],
				  [0,0,0]])
	
	landmark_noise = 10e-6 * np.eye(3)
	landmark_mean = np.zeros((3*feats.shape[1],1))
	landmark_seen = np.zeros((feats.shape[1],1))
	landmark_cov = np.zeros((3*feats.shape[1],3*feats.shape[1]))

	for i in range(feats.shape[1]):
		landmark_cov[3*i:3*i+3,3*i:3*i+3] = landmark_noise #initialize covariance

	count = 0
	IxV = 10e-6 * np.eye(4*feats.shape[1])

	H_t = np.zeros((4*feats.shape[1],3*feats.shape[1]))

	for i,tau_i in enumerate(tqdm.tqdm(np.squeeze(tau))):
		mew_odometry.append( mew_odometry[-1] @ expm(tau_i * zeta_hat[i]) )
		cov_odometry.append(expm(-tau_i*zeta_pointy_hat[i]) @ cov_odometry[-1] @ expm(-tau_i*zeta_pointy_hat[i]).T \
					  + W)
		
		available_feats_indices = np.nonzero(
			np.apply_along_axis(lambda x: np.all(x != null_condition),0,feats[:,:,i])
			)[0]
		
		cam_T_world = cam_T_imu @ mew_odometry[-1] #mew_odometry[-1] is imu_T_world
		world_T_cam = mew_odometry[-1] @ imu_T_cam
		z = feats[:,:,i]
		z_tilda = feats[:,:,i]
		for j in available_feats_indices:
			count += 1
			H_t = np.zeros((4*feats.shape[1],3*feats.shape[1]))

			if landmark_seen[j] == 0:
				
				landmark_seen[j] = 1
				d = feats[0,j,i] - feats[2,j,i]

				Z_0 = (K[0,0] * b) / d

				camera_coords = np.hstack((Z_0 * np.linalg.inv(K) @ np.hstack((feats[:2,j,i], 1)), 1))
				
				landmark_local = world_T_cam @ np.array([camera_coords]).T

				landmark_mean[(3*j):(3*j+3)] = landmark_local[:3]
				
			else:
				landmark_local = cam_T_world @ np.vstack([landmark_mean[(3*j):(3*j+3)],1])
				z_tilda[:,j] = np.squeeze(K_s @ projection(landmark_local.T).T)
				
				H_t[4*j:4*j+4,3*j:3*j+3] = (K_s @ projectionJacobian(landmark_local.T) @ cam_T_world @ P)[0] #4x3
				
			
			
			K_t = landmark_cov @ H_t.T @ \
				(H_t @ landmark_cov @ H_t.T + IxV)
			
			landmark_mean = landmark_mean + \
				np.array([K_t @ ( z - z_tilda ).T.flatten()]).T
			
			landmark_cov = (np.eye(3*feats.shape[1]) - K_t @ H_t) @ landmark_cov
	
	visualize_trajectory_2d_scatter(np.transpose(mew_odometry,axes=(1,2,0)),
						 landmark_mean[0::3],landmark_mean[1::3],
						 path_name=f"Visual Update for dataset {dataset}",show_ori=True,
						 dataset=dataset,fname="visual_update_only")'''
	
	# (c) Visual-Inertial SLAM
	mew_odometry = [T_0]
	W = np.diag([.01,.01,.01,.005,.005,.005])
	cov_odometry = [W]
	null_condition = np.array([-1,-1,-1,-1])

	P = np.array([[1,0,0],
			   	  [0,1,0],
				  [0,0,1],
				  [0,0,0]])
	
	landmark_noise = 10e-6 * np.eye(3)
	landmark_mean = np.zeros((3*feats.shape[1],1))
	landmark_seen = np.zeros((feats.shape[1],1))
	landmark_cov = np.zeros((3*feats.shape[1],3*feats.shape[1]))

	for i in range(feats.shape[1]):
		landmark_cov[3*i:3*i+3,3*i:3*i+3] = landmark_noise #initialize covariance

	count = 0
	IxV = 10e-6 * np.eye(4*feats.shape[1])

	H_t = np.zeros((4*feats.shape[1],3*feats.shape[1]))

	for i,tau_i in enumerate(tqdm.tqdm(np.squeeze(tau))):
		mew_odometry.append( mew_odometry[-1] @ expm(tau_i * zeta_hat[i]) )
		cov_odometry.append(expm(-tau_i*zeta_pointy_hat[i]) @ cov_odometry[-1] @ expm(-tau_i*zeta_pointy_hat[i]).T \
					  + W)
		
		available_feats_indices = np.nonzero(
			np.apply_along_axis(lambda x: np.all(x != null_condition),0,feats[:,:,i])
			)[0]
		
		cam_T_world = cam_T_imu @ mew_odometry[-1] #mew_odometry[-1] is imu_T_world
		world_T_cam = mew_odometry[-1] @ imu_T_cam
		z = feats[:,:,i]
		z_tilda = feats[:,:,i]
		for j in available_feats_indices:
			count += 1
			H_t = np.zeros((4*feats.shape[1],3*feats.shape[1]))

			if landmark_seen[j] == 0:
				
				landmark_seen[j] = 1
				d = feats[0,j,i] - feats[2,j,i]

				Z_0 = (K[0,0] * b) / d

				camera_coords = np.hstack((Z_0 * np.linalg.inv(K) @ np.hstack((feats[:2,j,i], 1)), 1))
				
				landmark_local = world_T_cam @ np.array([camera_coords]).T

				landmark_mean[(3*j):(3*j+3)] = landmark_local[:3]
				
			else:
				landmark_local = cam_T_world @ np.vstack([landmark_mean[(3*j):(3*j+3)],1])
				z_tilda[:,j] = np.squeeze(K_s @ projection(landmark_local.T).T)
				
				H_t[4*j:4*j+4,3*j:3*j+3] = (K_s @ projectionJacobian(landmark_local.T) @ cam_T_world @ P)[0] #4x3
				
			
			
			K_t = landmark_cov @ H_t.T @ \
				(H_t @ landmark_cov @ H_t.T + IxV)
			
			landmark_mean = landmark_mean + \
				np.array([K_t @ ( z - z_tilda ).T.flatten()]).T
			
			landmark_cov = (np.eye(3*feats.shape[1]) - K_t @ H_t) @ landmark_cov
	
	visualize_trajectory_2d_scatter(np.transpose(mew_odometry,axes=(1,2,0)),
						 landmark_mean[0::3],landmark_mean[1::3],
						 path_name=f"Visual Update for dataset {dataset}",show_ori=True,
						 dataset=dataset,fname="visual_update_only")


	# You may use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)
