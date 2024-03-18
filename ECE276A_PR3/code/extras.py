# (c) Visual-Inertial SLAM
	mew_odometry = [T_0]
	mew_t_1_t = [T_0]
	W = np.diag([.01,.01,.01,.005,.005,.005])
	cov_odometry = [W]
	cov_t_1_t_odometry = [W]
	null_condition = np.array([-1,-1,-1,-1])

	P = np.array([[1,0,0],
			   	  [0,1,0],
				  [0,0,1],
				  [0,0,0]])
	
	#landmark_cov = 10e-6 * np.eye(3)
	landmark_mean = np.zeros((3*feats.shape[1],1))
	landmark_seen = np.zeros((feats.shape[1],1))

	count = 0
	IxV = 10e-2 * np.eye(4*feats.shape[1])

	H_t = np.zeros((4*feats.shape[1],6))

	for i,tau_i in enumerate(tqdm.tqdm(np.squeeze(tau))):
		mew_t_1_t.append( mew_odometry[-1] @ expm(tau_i * zeta_hat[i]) )
		cov_t_1_t_odometry.append(expm(-tau_i*zeta_pointy_hat[i]) @ cov_odometry[-1] @ expm(-tau_i*zeta_pointy_hat[i]).T \
					  + W)
		
		available_feats_indices = np.nonzero(
			np.apply_along_axis(lambda x: np.all(x != null_condition),0,feats[:,:,i])
			)[0]
		
		cam_T_world = cam_T_imu @ mew_t_1_t[-1] #mew_odometry[-1] is imu_T_world
		world_T_cam = mew_t_1_t[-1] @ imu_T_cam
		z = feats[:,:,i]
		z_tilda = feats[:,:,i]
		for j in available_feats_indices:
			count += 1
			H_t = np.zeros((4*feats.shape[1],6))

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

				funky = mew_t_1_t[-1] @ np.vstack([landmark_mean[(3*j):(3*j+3)],1])
				#print(dot_hat(funky).shape)
				H_t[4*j:4*j+4,:] = (-K_s @ projectionJacobian(landmark_local.T) @ cam_T_imu @ dot_hat(funky))[0] #4x6

				#print(H_t[4*j:4*j+4,:].shape)
			#print(H_t.shape)
			
			K_t = cov_t_1_t_odometry[-1] @ H_t.T @ np.linalg.pinv(H_t @ cov_t_1_t_odometry[-1] @ H_t.T + IxV)
			
			mew_odometry.append(mew_t_1_t[-1] @ expm( 
				axangle2twist(K_t @ ( z - z_tilda ).T.flatten())
				))
			
			cov_odometry.append( (np.eye(6) - K_t @ H_t) @ cov_t_1_t_odometry[-1] )
			
			
	visualize_trajectory_2d_scatter(np.transpose(mew_odometry,axes=(1,2,0)),
						 landmark_mean[0::3],landmark_mean[1::3],
						 path_name=f"Visual inertial slam for dataset {dataset}",show_ori=True,
						 dataset=dataset,fname="visual_inertial_slam")