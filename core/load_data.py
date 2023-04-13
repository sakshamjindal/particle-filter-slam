import numpy as np
import os

def load_data(data_dir, dataset_num):

  encoder_path = os.path.join(data_dir, "Encoders%d.npz"%dataset_num)
  hokuyo_path = os.path.join(data_dir, "Hokuyo%d.npz"%dataset_num)
  imu_path = os.path.join(data_dir, "Imu%d.npz"%dataset_num)
  kinect_path = os.path.join(data_dir, "Kinect%d.npz"%dataset_num)

  with np.load(encoder_path) as data:
    encoder_counts = data["counts"]
    encoder_stamps = data["time_stamps"]

  with np.load(hokuyo_path) as data:
    lidar_angle_min = data["angle_min"]
    lidar_angle_max = data["angle_max"]
    lidar_angle_increment = data["angle_increment"]
    lidar_range_min = data["range_min"]
    lidar_range_max = data["range_max"]
    lidar_ranges = data["ranges"]
    lidar_stamps = data["time_stamps"]

  with np.load(imu_path) as data:
    imu_angular_velocity = data["angular_velocity"]
    imu_linear_acceleration = data["linear_acceleration"]
    imu_stamps = data["time_stamps"]

  with np.load(kinect_path) as data:
    disp_stamps = data["disparity_time_stamps"]
    rgb_stamps = data["rgb_time_stamps"]

  return {
    "encoder_counts": encoder_counts,
    "encoder_stamps": encoder_stamps,
    "lidar_angle_min": lidar_angle_min,
    "lidar_angle_max": lidar_angle_max,
    "lidar_angle_increment": lidar_angle_increment,
    "lidar_range_min": lidar_range_min,
    "lidar_range_max": lidar_range_max,
    "lidar_ranges": lidar_ranges,
    "lidar_stamps": lidar_stamps,
    "imu_angular_velocity": imu_angular_velocity,
    "imu_linear_acceleration": imu_linear_acceleration,
    "imu_stamps": imu_stamps,
    "disp_stamps": disp_stamps,
    "rgb_stamps": rgb_stamps
  }
if __name__ == '__main__':
  dataset = 20
  
  data_dir = os.path.join(os.getcwd(), "data")
  data = load_data(data_dir, dataset)
  print(data.keys())

  encoder_counts = data["encoder_counts"]
  encoder_stamps = data["encoder_stamps"]
  lidar_angle_min = data["lidar_angle_min"]
  lidar_angle_max = data["lidar_angle_max"]
  lidar_angle_increment = data["lidar_angle_increment"]
  lidar_range_min = data["lidar_range_min"]
  lidar_range_max = data["lidar_range_max"]
  lidar_ranges = data["lidar_ranges"]
  lidar_stamps = data["lidar_stamps"]
  imu_angular_velocity = data["imu_angular_velocity"]
  imu_linear_acceleration = data["imu_linear_acceleration"]
  imu_stamps = data["imu_stamps"]
  disp_stamps = data["disp_stamps"]
  rgb_stamps = data["rgb_stamps"]
    

