# contains data loaders for lidar, encoder and kinect

import numpy as np
from .points import LidarPoints
from .model import load_motion_model

def sync_data(original_data, original_stamps, target_stamps):
    index = np.argmin(np.abs(original_stamps - target_stamps))
    return original_data[index]

class LidarLoader():
    def __init__(self, data):
        self.lidar_keys = [key for key in data.keys() if "lidar" in key]

        self.lidar_stamps = data["lidar_stamps"].T
        self.lidar_ranges = data["lidar_ranges"].T
        self.lidar_angle_min = data["lidar_angle_min"].T
        self.lidar_angle_max = data["lidar_angle_max"].T
        self.lidar_angle_increment = data["lidar_angle_increment"]
        self.lidar_angles_num = self.lidar_ranges.shape[1]
        self.lidar_angles = np.linspace(self.lidar_angle_min, self.lidar_angle_max, self.lidar_angles_num)

    def __len__(self):
        return len(self.lidar_stamps)
    
    def __call__(self, i):
        return LidarPoints(self.lidar_ranges[i], self.lidar_angles)
    
class EncoderIMULoader():
    def __init__(self, data):

        self.encoder_counts = data["encoder_counts"].T
        self.encoder_stamps = data["encoder_stamps"].T

        self.imu_angular_velocity = data["imu_angular_velocity"].T
        self.imu_linear_acceleration = data["imu_linear_acceleration"].T
        self.imu_stamps = data["imu_stamps"].T

    def motion_model(self, index):
        encoder_count = self.encoder_counts[index]
        dt = self.encoder_stamps[index] - self.encoder_stamps[index - 1]
        omega = sync_data(self.imu_angular_velocity, self.imu_stamps, self.encoder_stamps[index])
        yaw_rate = omega[2]
        return load_motion_model(encoder_count, yaw_rate, dt)
    
    def __repr__(self):
        return f"Encoder: {self.data.shape}"
    

class DisparityLoader():
    def __init__(self, data):
        self.disp_stamps = data["disp_stamps"].T

    def __len__(self):
        return len(self.disp_stamps)

    def __call__(self, i):
        return self.disp_stamps[i]
    
class RGBLoader():
    def __init__(self, data):
        self.rgb_stamps = data["rgb_stamps"].T

    def __len__(self):
        return len(self.rgb_stamps)

    def __call__(self, i):
        return self.rgb_stamps[i]
