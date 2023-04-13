# contains robot class and related functions

import numpy as np
from .map import Map, ParticleMap
from .points import LidarPoints
import matplotlib.pyplot as plt
from .params import *

# from .utils import parallize

def softmax(x):
    x = x - x.max()
    return(np.exp(x)/np.exp(x).sum())

def body_T_optical():

    R_roll = np.array([
        [1, 0, 0], 
        [0, np.cos(roll), -np.sin(roll)], 
        [0, np.sin(roll), np.cos(roll)]
    ])
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = R_roll @ R_pitch @ R_yaw

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t_kinect

    return T
         
class Robot():
    def __init__(self, N = 100, sigma = 0, parallize = True, dataset_dir = "./"):
        self.initialise_particles(N)
        self.N = N
        self.parallize = parallize
        self.Neff = N/10.0
        self.sigma = sigma
        self.datset_dir = dataset_dir
        self.map = Map(self.datset_dir)

    def initialise_particles(self, N):
        self.particles = [Particle(0, 0, 0, 1.0/N) for _ in range(N)]

        self.pose = (self.particles[0]).pose()
        self.state = (self.particles[0]).get_state()
        self.weights = np.array([particle.weight for particle in self.particles])
        self.state_history = np.array([self.state])
        self.weights_history = np.array([self.weights])

    def predict(self, motion_model):
        for particle in self.particles:
            motion_model.apply(particle, sigma = self.sigma)

    def _update_robot_state(self):
        best_particle = self.particles[np.argmax(self.weights)]

        self.pose = best_particle.pose()
        self.state = best_particle.get_state()
        self.state_history = np.vstack((self.state_history, self.state))
        self.weights_history = np.vstack((self.weights_history, self.weights))

    def _resample(self):
        # stratified resampling
        #print("Resampling...")
        samples = np.random.choice(self.particles, size=self.N, replace=True, p=self.weights)
        new_particles = []

        for sample in samples:
            new_particles.append(Particle(sample.x, sample.y, sample.theta, 1.0/self.N))

        self.particles = new_particles

    def get_correlations(self, lidar_points, particles):
        # corrs = []
        # temporary_map = ParticleMap(self.map.map)

        # for particle in particles:
        #     corrs.append(temporary_map.calculate_correlation(lidar_points, particle.pose()))

        temporary_map = ParticleMap(self.map.map)

        from joblib import Parallel, delayed
        def process(a, b):
            return temporary_map.calculate_correlation(a, b)
        
        if self.parallize:
            corrs = Parallel(n_jobs=-1, prefer="threads")(delayed(process)(lidar_points, particle.pose()) for particle in particles)
        else:
            corrs = [temporary_map.calculate_correlation(lidar_points, particle.pose()) for particle in particles]

        return corrs

    def update(self, lidar_points, resample = True):
        corrs = []

        corrs = self.get_correlations(lidar_points, self.particles)
        corrs = np.array(corrs)

        # calculate new weights
        self.weights = softmax(corrs)
        #print(self.weights.max())

        # update robot state with best particle
        self._update_robot_state()

        # set weights for each particle
        if resample:
            if (1.0/np.sum((self.weights**2))) <= self.Neff:
                #print(f"{1.0/np.sum((self.weights**2))} {np.max(self.weights.max())} Resampling...")
                self._resample()
            else:
                #print(f"{1.0/np.sum((self.weights**2))} No Resampling...")
                for particle, weight in zip(self.particles, self.weights):
                    particle.weight = weight  
        else:
            print(f"{np.sum((self.weights**2))} No Resampling...")
            for particle, weight in zip(self.particles, self.weights):
                particle.weight = weight

        # update map
        self.update_map(lidar_points)

    def update_map(self, lidar_points):
        self.map.update(lidar_points, self.pose)

    def plot_trajectory(self):

        trajectory_cells = self.map.points_to_cells(self.state_history[:, :2])
        trajectory_cells = self.map.get_valid_cells(trajectory_cells)

        plt.tight_layout()
        plt.plot(trajectory_cells[:, 0], trajectory_cells[:, 1], 'r-')
        # add title and axis names
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f"{self.datset_dir}/trajectory.png")
        # save figure
    
    def __repr__(self) -> str:
        return f"Robot: x = {self.state[0]}, y = {self.state[1]}, theta = {self.state[2]}"

class Particle():

    def __init__(self, x, y, theta, weight):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        self.history = np.array([x, y, theta])

    def predict(self, dx, dy, dtheta):
        self.x = self.x + dx
        self.y = self.y + dy
        self.theta = self.theta + dtheta
        self.history = np.vstack((self.history, np.array([self.x, self.y, self.theta])))

    def update(self, lidar_points, pose):
        pass

    def get_state(self):
        return self.x, self.y, self.theta
    
    def __repr__(self):
        return f"Particle: x = {self.x}, y = {self.y}, theta = {self.theta}"
    
    def pose(self):
        return np.array([
            [np.cos(self.theta), -np.sin(self.theta), self.x],
            [np.sin(self.theta), np.cos(self.theta), self.y],
            [0, 0, 1]
        ]).astype(np.float32).reshape(3, 3)


