import numpy as np
from .params import *

class MotionModel():

    def __init__(self, vt, wt, dt):
        self.vt = vt
        self.wt = wt
        self.dt = dt

    def apply(self, particle, sigma = 0):
        particle.predict(
            self.dt*self.vt*np.sinc(self.wt*self.dt/2)*np.cos(particle.theta + self.wt*self.dt/2) + np.random.randn()*sigma,
            self.dt*self.vt*np.sinc(self.wt*self.dt/2)*np.sin(particle.theta + self.wt*self.dt/2) + np.random.randn()*sigma,
            self.dt*self.wt + np.random.randn()*0.001
        )
    def __repr__(self):
        return f"v = {self.vt}, w = {self.wt}, dt = {self.dt}"

def load_motion_model(encoder_count, omega, dt):
    
    # encoder coiunts the rotation of the wheel at 40 Hz

    assert round(wheel_distance_per_tick, 4) == 0.0022

    FR = encoder_count[0]
    FL = encoder_count[1]
    RR = encoder_count[2]
    RL = encoder_count[3]

    right_wheel_distance = (FR + RR)/2 * 0.0022
    left_wheel_distance = (FL + RL)/2 * 0.0022

    v_r = right_wheel_distance/dt
    v_l = left_wheel_distance/dt

    v = (v_r + v_l)/2

    return MotionModel(v, omega, dt)
