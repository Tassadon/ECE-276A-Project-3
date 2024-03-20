import cupy as np
from pr3_utils import *
from scipy.linalg import expm
import tqdm
import matplotlib.pyplot as plt


if __name__ == '__main__':

	# Load the measurements
	dataset = "10"
	filename = f"../data/{dataset}.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
	feats = features[:,::50,:] #15 for dataset 03 and 50 for dataset 10
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

	print("pointy_hat_shape", zeta_pointy_hat[0].shape)
	print("features:", features.shape)
	print("K:", K.shape)
	print("b:", b)
	print(imu_T_cam.shape)
	print("linear_velocity:",linear_velocity.shape)
	print("angular velocity:", angular_velocity.shape)
	print("time:",t.shape)
	print("feats:", feats.shape)
	
	# (a) IMU Localization via EKF Prediction: no update
	T_0 = np.array([
					[1,0,0,0],
					[0,-1,0,0],
					[0,0,-1,0],
					[0,0,0,1]])
	#T_0 = np.eye(4)
	'''
	poses = [T_0]
	for i,tau_i in enumerate(tqdm.tqdm(np.squeeze(tau))):
		poses.append( poses[-1] @ expm(tau_i * zeta_hat[i]) )
		#The covariance doesnt affect mean here

	poses = np.array(poses)
	visualize_trajectory_2d(np.transpose(poses,axes=(1,2,0)),path_name=f"IMU prediction only for dataset {dataset}",show_ori=True,
						 dataset=dataset,fname="inertial_prediction_only")
	
	# (b) Landmark Mapping via EKF Update: no prediction
	
	mew_odometry = [T_0]
	W = np.diag([.01,.01,.01,.005,.005,.005])
	cov_odometry = [W]
	null_condition = np.array([-1,-1,-1,-1])

	P = np.array([[1,0,0],
			   	  [0,1,0],
				  [0,0,1],
				  [0,0,0]])
	
	landmark_noise = 5 * np.eye(3)
	landmark_mean = np.zeros((3*feats.shape[1],1))
	landmark_seen = np.zeros((feats.shape[1],1))
	landmark_cov = np.zeros((3*feats.shape[1],3*feats.shape[1]))
	for i in range(feats.shape[1]):
		landmark_cov[3*i:3*i+3,3*i:3*i+3] = landmark_noise #initialize covariance
	IxV = 3 * np.eye(4*feats.shape[1])

	H_t = np.zeros((4*feats.shape[1],3*feats.shape[1]))
	count = 0
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
				
			#print("H_t", np.unique(H_t))
			
		landmark_cov = (landmark_cov + landmark_cov.T)/2
		K_t = landmark_cov @ H_t.T @ np.linalg.inv(H_t @ landmark_cov @ H_t.T + IxV)
		#print("K_t", np.unique(K_t))
		landmark_mean = landmark_mean + \
			np.array([K_t @ ( z - z_tilda ).T.flatten()]).T
			
		g = K_t @ H_t
			
		#print(np.unique(g))
		landmark_cov = (np.eye(3*feats.shape[1]) - g) @ landmark_cov
			
			
	visualize_trajectory_2d_scatter(np.transpose(mew_odometry,axes=(1,2,0)),
						 landmark_mean[0::3],landmark_mean[1::3],
						 path_name=f"Visual Update for dataset {dataset}",show_ori=True,
						 dataset=dataset,fname="visual_update_only")
	'''
	# (c) Visual-Inertial SLAM

	mew_odometry = [T_0]
	null_condition = np.array([-1,-1,-1,-1])

	P = np.array([[1,0,0],
			   	  [0,1,0],
				  [0,0,1],
				  [0,0,0]])
	
	
	W = np.diag([1,1,1,.5,.5,.5])
	count = 0
	IxV_L = 1 * np.eye(4*feats.shape[1])
	IxV_M = 5 * np.eye(4*feats.shape[1])
	landmark_noise = 3 * np.eye(3)
	
	H_t = np.zeros((4*feats.shape[1],3*feats.shape[1] + 6))
	H_L = np.zeros((4*feats.shape[1],6))
	H_M = np.zeros((4*feats.shape[1],3*feats.shape[1]))
	slam_cov = np.zeros((3*feats.shape[1]+6,3*feats.shape[1]+6))

	for i in range(feats.shape[1]):
		slam_cov [3*i:3*i+3,3*i:3*i+3] = landmark_noise #initialize covariance
	slam_cov[-6:,-6:] = W

	landmark_mean = np.zeros((3*feats.shape[1],1))
	landmark_seen = np.zeros((feats.shape[1],1))
	mew_t = T_0
	for i,tau_i in enumerate(tqdm.tqdm(np.squeeze(tau))):
		mew_t = ( mew_odometry[-1] @ expm(tau_i * zeta_hat[i]) )

		F = expm(-tau_i*zeta_pointy_hat[i])

		slam_cov[-6:,-6:] = F @ slam_cov[-6:,-6:] @ F.T + W

		slam_cov[-6:,:-6] = F @ slam_cov[-6:,:-6]
		slam_cov[:-6,-6:] = slam_cov[:-6,-6:] @ F.T
		

		available_feats_indices = np.nonzero(
			np.apply_along_axis(lambda x: np.all(x != null_condition),0,feats[:,:,i])
			)[0]
		
		cam_T_world = cam_T_imu @ mew_t #mew_odometry[-1] is imu_T_world
		world_T_cam = mew_t @ imu_T_cam
		z = feats[:,:,i]
		z_tilda = feats[:,:,i]

		for j in available_feats_indices:
			count += 1
			H_L = np.zeros((4*feats.shape[1],6))
			H_M = np.zeros((4*feats.shape[1],3*feats.shape[1]))

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

				funky = mew_t @ np.vstack([landmark_mean[(3*j):(3*j+3)],1])

				H_L[4*j:4*j+4,:] = (-K_s @ projectionJacobian(landmark_local.T) @ cam_T_imu @ dot_hat(funky))[0] #4x6

				H_M[4*j:4*j+4,3*j:3*j+3] = (K_s @ projectionJacobian(landmark_local.T) @ cam_T_world @ P)[0] #4x3
		

		H_t = np.hstack([H_M,H_L]) # 4N x (3M + 6)
		
		K_t = slam_cov @ H_t.T @ np.linalg.pinv(H_t @ slam_cov @ H_t.T + IxV_L) # (3M + 6) x 4N
		#print("feats",feats.shape[1])
		#print(K_t.shape)
		
		mew_odometry.append(mew_t @ expm(
				axangle2twist(K_t[-6:] @ ( z - z_tilda ).T.flatten())
				))
			
		landmark_mean = landmark_mean + np.array([K_t[:-6,:] @ ( z - z_tilda ).T.flatten()]).T

		slam_cov = (np.eye(3*feats.shape[1]+6) - K_t @ H_t) @ slam_cov
			
	visualize_trajectory_2d_scatter(np.transpose(mew_odometry,axes=(1,2,0)),
						 landmark_mean[0::3],landmark_mean[1::3],
						 path_name=f"Visual inertial slam for dataset {dataset}",show_ori=True,
						 dataset=dataset,fname="visual_inertial_slam")
	
	# You may use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)
