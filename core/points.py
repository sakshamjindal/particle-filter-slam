import numpy as np


class Points():
    def __init__(self, points):
        # points:(N, X)
        self.points = points
        assert self.points.shape[1] == 2 or self.points.shape[1] == 3

    def __len__(self):
        return len(self.points)
    
    @classmethod
    def homogeneous(cls, points):
        # Add a column of ones to the points
        # return a point instance of homogeneous coordinates
        return cls(np.hstack([points, np.ones((len(points), 1))]))
    
    @property
    def H(self):
        return self.homogeneous(self.points)
    
    @classmethod
    def unhomogeneous(cls, points):
        # Remove the last column of ones
        # return a point instance of homogeneous coordinates
        return cls(points[:, :-1])
    
    @property
    def unH(self):
        return self.unhomogeneous(self.points)
    
    @property
    def pts(self):
        return self.points
    
    def __len__(self):
        return len(self.points)
    
    
class LidarPoints():
    def __init__(self, lidar_range, lidar_angle):

        self.lidar_range = lidar_range
        self.lidar_angle = lidar_angle

        self.points = Points(self.cartesian_points())

    def cartesian_points(self):
        return (np.vstack(
            [self.lidar_range * np.cos(self.lidar_angle), self.lidar_range * np.sin(self.lidar_angle)]
        ).T)
    
    def __len__(self):
        return len(self.points)