import numpy as np
from tqdm import tqdm
import os

from core.robot import Robot
from core.dataloader import EncoderIMULoader, LidarLoader
from core.map import Map
from core.load_data import load_data
import matplotlib.pyplot as plt


experiments = {
    "N": [100, 1000],
    "lidar": [True],
    "sigma": [0.01, 0.05]
}


os.makedirs("results2", exist_ok=True)
for N in experiments["N"]:
    for lidar in experiments["lidar"]:
        for sigma in experiments["sigma"]:
            print(f"N: {N}, lidar: {lidar}, sigma: {sigma}")

            if N==1000 and sigma==0.001:
                continue

            save_dir = f"results2/N_{N}_lidar_{lidar}_sigma_{sigma}"
            os.makedirs(save_dir, exist_ok=True)

            dataset_num = 20
            dataset_dir = "data"
            data = load_data(dataset_dir, dataset_num)

            robot = Robot(N = N, sigma = sigma, parallize=False, dataset_dir=save_dir)

            encoder = EncoderIMULoader(data)
            lidar_data = LidarLoader(data)

            encoder_stamps = encoder.encoder_stamps
            lidar_stamps = lidar_data.lidar_stamps

            timestamps = np.concatenate((encoder_stamps, lidar_stamps))
            timestamps.sort()

            for index, timestamp in tqdm(enumerate(timestamps)):
                if timestamp in encoder_stamps:
                    encoder_index = np.where(encoder_stamps == timestamp)[0][0]

                    if encoder_index == 0:
                        continue

                    motion_model = encoder.motion_model(encoder_index)
                    robot.predict(motion_model)

                elif timestamp in lidar_stamps:
                    lidar_index = np.where(lidar_stamps == timestamp)[0][0]
                    lidar_points = lidar_data(lidar_index)

                    robot.update(lidar_points.points)
                else:
                    print("Something went wrong")

            #plot trajectory
            robot.plot_trajectory()

            # plot map
            robot.map.plot()

            import pickle
            with open(f"{save_dir}/robot.pickle", "wb") as f:
                pickle.dump(robot, f)

            with open(f"{save_dir}/map.pickle", "wb") as f:
                pickle.dump(robot.map, f)

            del robot