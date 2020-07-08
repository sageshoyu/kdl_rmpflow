import time
import copy
import pathlib
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer

class MjJacoEnv(object):
    """docstring for MjJacoEnv."""

    def __init__(self, vis=False):
        super(MjJacoEnv, self).__init__()
        #parent_dir_path = str(pathlib.Path(__file__).parent.absolute())
        # self.fname = parent_dir_path + '/jaco/jaco.xml'
        self.fname = 'assets/kinova_j2s6s300/mj-j2s6s300.xml'
        self.model = load_model_from_path(self.fname)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.vis=vis

    def step(self, action):
        for i in range(len(action)):
            self.sim.data.ctrl[i]=action[i]

        self.sim.forward()
        self.sim.step()
        self.viewer.render() if self.vis else None
        return self.sim.data.qpos
