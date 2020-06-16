import copy
from scipy.spatial.transform import Rotation as R
import numpy as np
import motor_skills.core.mj_control as mjc
from motor_skills.envs.mj_jaco import MjJacoEnv
from motor_skills.rmp.rmp import RMPRoot
from motor_skills.rmp.kdl_rmp import KDLRMPNode
from motor_skills.rmp.kdl_rmp import ProjectionNode
from urdf_parser_py.urdf import URDF as u_parser
from motor_skills.rmp.kdl_rmp import rmp_from_urdf, PositionProjection, kdl_node_array
import motor_skills.rmp.rmp_leaf as leaves

# %%
env = MjJacoEnv(vis=True)

# set the jaco arm to a stable(ish) position
env.sim.data.qpos[:12] = [0, np.pi, np.pi, 0, np.pi, 0, 0, 0, 0, 0, 0, 0]
# env.sim.data.qpos[:6] = [2.5, 1, 1, 1, 1, 1]

env.sim.data.qvel[:6] = [0,0,0,0,0,0]

r_xpos = np.size(env.sim.data.body_xpos, 0)
target_pos = env.sim.data.body_xpos[r_xpos - 3]
obstacle_pos = env.sim.data.body_xpos[r_xpos - 2]

# load URDF
robot = u_parser.from_xml_file('assets/kinova_j2s6s300/ros-j2s6s300.xml')
root, links = rmp_from_urdf(robot)

link5_pos = PositionProjection("link5_pos", links['j2s6s300_link_5'])
link6_pos = PositionProjection("link6_pos", links['j2s6s300_link_6'])

link6_proj = ProjectionNode("link6_proj", root, np.array([1]*6 + [0]*6))
link6_exts = kdl_node_array("link6_ext", link6_proj, robot, 'world', 'j2s6s300_link_6',
                            spacing=0.2, skip=1, num=3, link_dir=np.array([0, 0, -1]).reshape(-1, 1))
link6_exts_pos = [PositionProjection(link6_ext.name + "_pos", link6_ext) for link6_ext in link6_exts]

fing1_pos = PositionProjection("fing1_pos", links['j2s6s300_link_finger_1'])
fingtip1_pos = PositionProjection("fingtip1_pos", links['j2s6s300_link_finger_tip_1'])

atrc = leaves.GoalAttractorUni("jaco_attractor", link6_exts_pos[0], np.array([target_pos]).T, gain=20)

# obst0 = leaves.CollisionAvoidance("jaco_avoider0", link5_pos, None,
#                                   np.array([obstacle_pos]).T, R=0.05, eta=3, epsilon=0.0)
# obst1 = leaves.CollisionAvoidance("jaco_avoider1", link6_pos, None,
#                                   np.array([obstacle_pos]).T, R=0.05, eta=3, epsilon=0.0)
# obst2 = leaves.CollisionAvoidance("jaco_avoider2", fing1_pos, None,
#                                   np.array([obstacle_pos]).T, R=0.05, eta=3, epsilon=0.0)
# obst3 = leaves.CollisionAvoidance("jaco_avoider3", fingtip1_pos, None,
#                                   np.array([obstacle_pos]).T, R=0.05, eta=3, epsilon=0.0)

jnts = ['j2s6s300_joint_1',
        'j2s6s300_joint_2',
        'j2s6s300_joint_3',
        'j2s6s300_joint_4',
        'j2s6s300_joint_5',
        'j2s6s300_joint_6',
        'j2s6s300_joint_finger_1',
        'j2s6s300_joint_finger_tip_1',
        'j2s6s300_joint_finger_2',
        'j2s6s300_joint_finger_tip_2',
        'j2s6s300_joint_finger_3',
        'j2s6s300_joint_finger_tip_3']

def get_lims(name):
    jntlim = robot.joint_map[name].limit
    return [jntlim.lower, jntlim.upper]


lims = np.array(list(map(get_lims, jnts)))
cent = np.mean(lims, axis=1).reshape(-1, 1)
jnt_lim = leaves.JointLimiter("jaco_jnt_lims", root, lims, cent, lam=0.01)


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
