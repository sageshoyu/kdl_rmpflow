import copy
from scipy.spatial.transform import Rotation as R
import numpy as np
import motor_skills.core.mj_control as mjc
from motor_skills.envs.mj_jaco import MjJacoEnv
from motor_skills.rmp.rmp import RMPRoot
from motor_skills.rmp.kdl_rmp import KDLFlatRMPNode
import motor_skills.rmp.rmp_leaf as leaves

# %%
env = MjJacoEnv(vis=True)

# set the jaco arm to a stable(ish) position
env.sim.data.qpos[:6] = [2.5, 1, 1, 1, 1, 1]
env.sim.data.qvel[:6] = [0]*6

r_xpos = np.size(env.sim.data.body_xpos, 0)
target_pos = env.sim.data.body_xpos[r_xpos - 3]
obstacle_pos = env.sim.data.body_xpos[r_xpos - 2]

root = RMPRoot("jaco_root")
hand = KDLFlatRMPNode("jaco_thumb", root,
                      'assets/kinova_j2s6s300/ros-j2s6s300.xml',
                      'world',
                      'j2s6s300_link_finger_tip_1')
atrc = leaves.GoalAttractorUni("jaco_attractor", hand,
                               np.array([target_pos]).T, gain=3.5)
obst = leaves.CollisionAvoidanceSphere("jaco_avoider", hand, None,
                                       np.array([obstacle_pos]).T, R=0.05)

qdd_cap = 1000
while True:

    # evaluate RMP for goal
    q = env.sim.data.qpos[:8]
    qd = env.sim.data.qvel[:8]
    qdd = root.solve(np.array([q]).T, np.array([qd]).T).flatten().tolist()
    action = mjc.pd(qdd, qd, q, env.sim, ndof=8)

    action_norm = np.linalg.norm(action)
    if action_norm > qdd_cap:
        action = action / action_norm * qdd_cap

    try:
        env.step(action)
    except:
        print("bad qdd: " + str(qdd))
        break

    #print('qpos: ' + str(env.sim.data.qpos[:6]))
    print('xpos: ' + str(env.sim.data.body_xpos[8]))
    #quat = mjc.quat_to_scipy(env.sim.data.body_xquat[6])
    #r = R.from_quat(quat)
    #print('rot: ' + str(r.as_euler('xyz', degrees=False)))
    #print('qdd: ' + str(qdd))
