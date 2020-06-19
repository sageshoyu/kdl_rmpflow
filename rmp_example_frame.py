import numpy as np
import motor_skills.core.mj_control as mjc
from motor_skills.envs.mj_jaco import MjJacoEnv
from motor_skills.rmp.kdl_rmp import ProjectionNode
from urdf_parser_py.urdf import URDF as u_parser
from motor_skills.rmp.kdl_rmp import rmp_from_urdf, PositionProjection, KDLRMPNode, kdl_node_array
import motor_skills.rmp.rmp_leaf as leaves

# %%
env = MjJacoEnv(vis=True)

# set the jaco arm to a stable(ish) position
env.sim.data.qpos[:12] = [0, np.pi, np.pi, 0, np.pi, 0, 0, 0, 0, 0, 0, 0]
# env.sim.data.qpos[:6] = [3.5, 3, 1, 1, 1, 1]

env.sim.data.qvel[:6] = [0, 0, 0, 0, 0, 0]

r_xpos = np.size(env.sim.data.body_xpos, 0)
target_pos = env.sim.data.body_xpos[r_xpos - 5]
fright_pos = env.sim.data.body_xpos[r_xpos - 4]
fleft_pos = env.sim.data.body_xpos[r_xpos - 3]
fbot_pos = env.sim.data.body_xpos[r_xpos - 2]
ftop_pos = env.sim.data.body_xpos[r_xpos - 1]

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

# load URDF
robot = u_parser.from_xml_file('assets/kinova_j2s6s300/ros-j2s6s300.xml')

# construct basic kinematic RMP nodes
root, links = rmp_from_urdf(robot)

# attach position nodes
collision_pts_pos = {}
for name, node in links.items():
    collision_pts_pos[name] = PositionProjection(name + "_pos", node)

# add line of position nodes through hand to dictionary
link6_proj = ProjectionNode("link6_proj", root, np.array([1] * 6 + [0] * 6))
link6_exts = kdl_node_array("link6_ext", link6_proj, robot, 'world', 'j2s6s300_link_6',
                            spacing=0.05, skip=1, num=3, link_dir=np.array([0, 0, -1]).reshape(-1, 1))
link6_exts_pos = [PositionProjection(link6_ext.name + "_pos", link6_ext) for link6_ext in link6_exts]

# add to collision points position too
for hand_pos in link6_exts_pos:
    collision_pts_pos[hand_pos.name] = hand_pos


# attach collision avoidance nodes (avoid the frame)
for name, node in list(collision_pts_pos.items()):
    leaves.CollisionAvoidanceBox(name + "_right_avoider", node, None,
                                 np.array([fright_pos]).T, np.array([[0.05, 0.05, 0.10]]).T,
                                 0.005, epsilon=0.0, eta=0, r_w = 0.10)

    leaves.CollisionAvoidanceBox(name + "_left_avoider", node, None,
                                 np.array([fleft_pos]).T, np.array([[0.05, 0.05, 0.10]]).T,
                                 0.005, epsilon=0.0, eta=0, r_w = 0.10)

    leaves.CollisionAvoidanceBox(name + "_bot_avoider", node, None,
                                 np.array([fbot_pos]).T, np.array([[0.1, 0.05, 0.05]]).T,
                                 0.005, epsilon=0.0, eta=0, r_w = 0.10)

    leaves.CollisionAvoidanceBox(name + "_top_avoider", node, None,
                                 np.array([ftop_pos]).T, np.array([[0.1, 0.05, 0.05]]).T,
                                 0.005, epsilon=0.0, eta=0, r_w=0.10)

# attract palm of hand to target
atrc = leaves.GoalAttractorUni("jaco_attractor", link6_exts_pos[1], np.array([target_pos]).T, gain=20)

# include joint limits, biast towards center of each joint
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
        print("WARNING: HIT CAP")

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
