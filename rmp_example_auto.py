import copy
from scipy.spatial.transform import Rotation as R
import numpy as np
import motor_skills.core.mj_control as mjc
from motor_skills.envs.mj_jaco import MjJacoEnv
from motor_skills.rmp.rmp import RMPRoot
from motor_skills.rmp.kdl_rmp import KDLRMPNode
from motor_skills.rmp.kdl_rmp import ProjectionNode
from urdf_parser_py.urdf import URDF as u_parser
from motor_skills.rmp.kdl_rmp import tree_from_robot
import motor_skills.rmp.rmp_leaf as leaves

# %%
env = MjJacoEnv(vis=True)

# set the jaco arm to a stable(ish) position
env.sim.data.qpos[:12] = [0, np.pi, np.pi, 0, np.pi, 0, 1, 1, 1, 1, 0, 0]
# env.sim.data.qpos[:6] = [2.5, 1, 1, 1, 1, 1]

env.sim.data.qvel[:6] = [0,0,0,0,0,0]

r_xpos = np.size(env.sim.data.body_xpos, 0)
target_pos = env.sim.data.body_xpos[r_xpos - 3]
obstacle_pos = env.sim.data.body_xpos[r_xpos - 2]

# load URDF
robot = u_parser.from_xml_file('assets/kinova_j2s6s300/ros-j2s6s300.xml')
root, links = tree_from_robot(robot)

atrc = leaves.GoalAttractorUni("jaco_attractor", links['j2s6s300_link_finger_tip_1'], np.array([target_pos]).T, gain=20)

obst0 = leaves.CollisionAvoidance("jaco_avoider0", links['j2s6s300_link_5'], None,
                                  np.array([obstacle_pos]).T, R=0.05, eta=3, epsilon=0.0)
obst1 = leaves.CollisionAvoidance("jaco_avoider1", links['j2s6s300_link_6'], None,
                                  np.array([obstacle_pos]).T, R=0.05, eta=3, epsilon=0.0)
obst2 = leaves.CollisionAvoidance("jaco_avoider2", links['j2s6s300_link_finger_1'], None,
                                  np.array([obstacle_pos]).T, R=0.05, eta=3, epsilon=0.0)
obst3 = leaves.CollisionAvoidance("jaco_avoider3", links['j2s6s300_link_finger_tip_1'], None,
                                  np.array([obstacle_pos]).T, R=0.05, eta=3, epsilon=0.0)

qdd_cap = 1000
while True:
    # evaluate RMP for goal
    q = env.sim.data.qpos
    qd = env.sim.data.qvel
    qdd = root.solve(np.array([q]).T, np.array([qd]).T).flatten().tolist()
    action = mjc.pd(qdd, qd, q, env.sim, ndof=12)

    action_norm = np.linalg.norm(action)
    if action_norm > qdd_cap:
        action = action / action_norm * qdd_cap

    try:
        # print("action: " + str(action))
        env.step(action)
    except:
        print("bad qdd: " + str(qdd))
        break

    # print('qpos: ' + str(env.sim.data.qpos[:6]))
    print('xpos: ' + str(env.sim.data.body_xpos[7]))
    print('qpos: ' + str(env.sim.data.qpos))
    # quat = mjc.quat_to_scipy(env.sim.data.body_xquat[6])
    # r = R.from_quat(quat)
    # print('rot: ' + str(r.as_euler('xyz', degrees=False)))
    # print('qdd: ' + str(qdd))
