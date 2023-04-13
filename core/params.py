
import numpy as np

# data path
disp_path = "data/dataRGBD/Disparity20/"
rgb_path = "data/dataRGBD/RGB20/"

# map
MAP = {}
MAP['res']   = 0.05 #meters
MAP['xmin']  = -20  #meters
MAP['ymin']  = -20
MAP['xmax']  =  30
MAP['ymax']  =  30 
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['occ_conf'] = 0.8
MAP['max_logodd'] = 50
MAP['min_logodd'] = -50

# lidar
t_lidar = np.array([0, 0.030183/2])

# Kinect
roll, pitch, yaw = 0, 0.36, 0.021
t_kinect = np.array([0.18, 0.005, 0.36])

# motion model
wheel_diameter = 0.254
total_ticks_per_rotation = 360
wheel_distance_per_rotation = np.pi * wheel_diameter
wheel_distance_per_tick = wheel_distance_per_rotation/total_ticks_per_rotation



