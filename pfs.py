import numpy as np
from tqdm import tqdm

from core.robot import Robot
from core.dataloader import EncoderIMULoader, LidarLoader
from core.map import Map
from core.load_data import load_data


dataset_num = 20
dataset_dir = "data"

def run_pf_slam(dataset_dir, dataset_num, N, sigma):
    data = load_data(dataset_dir, dataset_num)

    robot = Robot(N = N, sigma = sigma, parallize=False)

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


    # plot trajectory
    robot.plot_trajectory()

    # plot map
    robot.map.plot()

if __name__ == '__main__':


    dataset_num = 20
    dataset_dir = "data"
    N = 1
    sigma = 0.1

    run_pf_slam(dataset_dir, dataset_num, N, sigma)

