
from .pr2_utils import bresenham2D
import numpy as np
from .points import Points
import matplotlib.pyplot as plt
from .params import *


class Map():
    def __init__(self, save_dir="./"):

        self.save_dir = save_dir

        # initialise variables from the MAP dictionary
        for key in MAP.keys():
            setattr(self, key, MAP[key])

        self._initialise_map()

    @property
    def body_T_lidar(self):
        
        R  = np.array(
            [[1, 0], 
             [0, 1]]
        )
        T = np.hstack([R, t_lidar.reshape(2, -1)])
        T = np.vstack([T, [0, 0, 1]])
        return T
    
    def lidar_to_world(self, lidar_points, pose):

        assert pose.shape == (3, 3)

        if not isinstance(lidar_points, Points):
            # give a warning if not
            raise TypeError("lidar_points is not a LidarPoints instance")

        body_T_lidar = self.body_T_lidar
        world_T_body = pose

        world_T_lidar = world_T_body @ body_T_lidar

        lidar_points = lidar_points.H.pts # array Nx3
        world_points = world_T_lidar @ lidar_points.T # array 3xN

        return Points(world_points.T).unH    

    def _initialise_map(self):
        # initialise the map
        self.map = np.zeros((self.sizex, self.sizey), dtype=np.float32)
        self.occ_logodd = np.log(self.occ_conf / (1 - self.occ_conf))
    
    @property
    def size(self):
        # return the size of the map
        return (self.sizex, self.sizey)
    
    def points_to_cells(self, points):
        # points: (N, 2)
        # convert from meters to cells
        assert points.shape[1] == 2

        xis = np.ceil((points[:, 0] - self.xmin) / self.res ).astype(np.int16)-1
        yis = np.ceil((points[:, 1] - self.ymin) / self.res ).astype(np.int16)-1
        return (np.vstack([xis, yis]).T).reshape(-1, 2)
    
    def get_valid_cells(self, cells):
        # check if the cells are within the map
        assert cells.shape[1] == 2

        valid = (cells[:, 0] >= 0) & (cells[:, 0] < self.sizex) & (cells[:, 1] >= 0) & (cells[:, 1] < self.sizey)
        cells = cells[valid, :]
        return cells
    
    def _update_log_odds(self, cells, logodd):
        # update the logodd of the cells

        # sanity check
        cells[cells[:, 0] < 0, 0] = 0
        cells[cells[:, 0] >= self.sizex, 0] = self.sizex - 1
        cells[cells[:, 1] < 0, 1] = 0
        cells[cells[:, 1] >= self.sizey, 1] = self.sizey - 1

        self.map[cells[:, 0], cells[:, 1]] += logodd

    def _prevent_logodd_overflow(self):
        # prevent overflow of the logodd
        self.map[self.map > self.max_logodd] = self.max_logodd
        self.map[self.map < self.min_logodd] = self.min_logodd 

    def update(self, lidar_points, pose):
        """Args:
            lidar_points: LidarPoints instance
            pose: (3, 3)
        Returns:
            None
        """

        if not isinstance(lidar_points, Points):
            # give a warning if not
            raise TypeError("lidar_points is not a LidarPoints instance")
        
        # for the particle, find position of particles in the map
        particle_cell = self.points_to_cells(pose[:2, -1].T.reshape(1, -1)).astype(np.int64)

        # for the lidar 
        world_points = self.lidar_to_world(lidar_points, pose)
        cells = self.points_to_cells(world_points.pts).astype(np.int64)
        #check and get if the cells are within the map
        cells = self.get_valid_cells(cells)

         # get the cells between the particle and the lidar
        particle_cells = np.repeat(particle_cell, len(cells), axis=0)
        ranges = np.hstack([particle_cells, cells]).astype(np.int64)

        # update the logodd of the cells
        for range in ranges:
            line = (bresenham2D(range).T).astype(np.int64)
            occupied = line[-1].reshape(1, -1)
            unoccupied = np.vstack([particle_cell, line[:-1, :]])
            self._update_log_odds(occupied, self.occ_logodd)
            self._update_log_odds(unoccupied, -self.occ_logodd)
            
        self._prevent_logodd_overflow()
        
        
    def plot(self):
        map_im  = np.zeros_like(self.map)
        map_im[self.map > 0] = -1
        map_im[self.map < 0] = 1
        
        # process the map to for imshow
        map_im = np.flip(map_im, axis=1)
        map_im = np.rot90(map_im)

        plt.tight_layout()
        plt.imshow(map_im, cmap='gray', origin='lower')
        # save figure
        plt.savefig(f"{self.save_dir}/map.png")
        plt.close()

class ParticleMap(Map):

    def __init__(self, robot_map):
        super().__init__()
        import copy
        self.robot_map = copy.deepcopy(robot_map)
        #robot map will have 0 for obstacles and 1 for free cells
        self.robot_map[self.robot_map > 0] = 0
        self.robot_map[self.robot_map < 0] = 1

    def get_valid_cells(self, cells):
        # check if the cells are within the map
        assert cells.shape[1] == 2
        valid = (cells[:, 0] >= 0) & (cells[:, 0] < self.sizex) & (cells[:, 1] >= 0) & (cells[:, 1] < self.sizey)
        return valid

    def calculate_correlation(self, lidar_point, pose):

        # particle position in the map
        particle_pos = pose[:2, -1].T.reshape(1, -1)
        # print(particle_pos.shape)
        
        # calculate lidar obstavle values in world frame
        world_points = self.lidar_to_world(lidar_point, pose) # (N, 2)
        # print(world_points.pts.shape)

        # sample points in the vicitnity of the lidar points
        x_range = np.arange(-0.2,0.2+0.05,0.05)
        y_range = np.arange(-0.2,0.2+0.05,0.05)
        x_range, y_range = np.meshgrid(x_range, y_range)
        sample_points = np.hstack([x_range.reshape(-1, 1), y_range.reshape(-1, 1)]) # (81, 2)

        # sample points for each world point
        sample_points = sample_points[np.newaxis, :, :] # (1, 81, 2)
        sample_points = np.repeat(sample_points, len(world_points), axis=0) # (N, 81, 2)

        # add world_points to sample_points
        sample_points = sample_points + world_points.pts[:, np.newaxis, :] # (N, 81, 2)

        # convert to cells
        sample_cells = self.points_to_cells(sample_points.reshape(-1, 2)) # (N*81, 2)
        sample_cells = sample_cells.reshape(-1, 81, 2)
        sample_cells = sample_cells.transpose(1, 0, 2) # (81, N, 2)

        # check if the cells are within the map
        valid = self.get_valid_cells(sample_cells.reshape(-1, 2)) # (N*81, 1)
        valid = valid.reshape(81, -1) # (81, N)
        valid_cells = [sample_cells[i][valid[i], :] for i in range(81)]

        # calculate correlation
        correlations = [np.sum(self.robot_map[valid_cells[i][:, 0], valid_cells[i][:, 1]]) for i in range(81)]
        return np.max(correlations)

# test
if __name__ == "__main__":

    data_path = "data"
    dataset_num = 20

    from utils import load_data
    from core.dataloader import LidarLoader
    data = load_data(data_path, dataset_num)
    lidar_data = LidarLoader(data)
    lidar = lidar_data(0)

    test_map = Map()
    pose = np.eye(3)
    # add some noise to the pose
    # update the map
    test_map.update(lidar.points, pose)
    # map_im = test_map.plot()

    particle_map = ParticleMap(test_map.map)
    particle_map.calculate_correlation(lidar.points, pose)



