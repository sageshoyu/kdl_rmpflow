import numpy as np
from scipy.spatial.transform import Rotation as R
import kdl_rmpflow.core.mj_control as mjc
from kdl_rmpflow.envs.mj_jaco import MjJacoEnv
from kdl_rmpflow.rmp.kdl_rmp import ProjectionNode
from urdf_parser_py.urdf import URDF as u_parser
from kdl_rmpflow.rmp.kdl_rmp import rmp_from_urdf, PositionProjection, KDLRMPNode, kdl_node_array, kdl_cylinder, RotZProjection, RotYProjection, RotXProjection
import kdl_rmpflow.rmp.rmp_leaf as leaves

#
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

# add cyl of collision points along forearm
link4_proj = ProjectionNode("link4_proj", root, np.array([1] * 4 + [0] * 8))
link4_cyls = kdl_cylinder("link4_rnd", link4_proj, robot, 'world', 'j2s6s300_link_4',
                          r=0.04, h=0.08, pts_per_round=3, pts_in_h=3, link_dir=np.array([0,0,-1]).reshape(-1,1))
link4_cyls_pos = [PositionProjection(link4_cyl.name + "_pos", link4_cyl) for link4_cyl in link4_cyls]
collision_pts_pos.update(dict(zip(
        map(lambda node: node.name, link4_cyls_pos),
        link4_cyls_pos)))

link5_proj = ProjectionNode("link5_proj", root, np.array([1] * 5 + [0] * 7))
link5_cyls = kdl_cylinder("link5_rnd", link5_proj, robot, 'world', 'j2s6s300_link_5',
                          r=0.04, h=0.10, pts_per_round=5, pts_in_h=4, link_dir=np.array([0,1,0]).reshape(-1,1))
link5_cyls_pos = [PositionProjection(link5_cyl.name + "_pos", link5_cyl) for link5_cyl in link5_cyls]
collision_pts_pos.update(dict(zip(
    map(lambda node: node.name, link5_cyls_pos),
    link5_cyls_pos)))

# add line of control points for attraction
link6_proj = ProjectionNode("link6_proj", root, np.array([1] * 6 + [0] * 6))
link6_exts = kdl_node_array("link6_ext", link6_proj, robot, 'world', 'j2s6s300_link_6',
                            h=0.15, skip_h=0.05, num=2, link_dir=np.array([0, 0, -1]).reshape(-1, 1))
link6_ext_rotx = RotXProjection("link6_rotx", link6_exts[1])
link6_ext_roty = RotYProjection("link6_roty", link6_exts[1])
link6_ext_rotz = RotZProjection("link6_rotz", link6_exts[1])
link6_exts_pos = [PositionProjection(link6_ext.name + "_pos", link6_ext) for link6_ext in link6_exts]

# add collision avoidance control points for hand base
link6_cyls = kdl_cylinder("link6_cyl", link6_proj, robot, 'world', 'j2s6s300_link_6',
                          r=0.05, h=0.10, pts_in_h=5, pts_per_round=6, link_dir=np.array([0,0,-1]).reshape(-1, 1))
link6_cyls_pos = [PositionProjection(link6_cyl.name + "_pos", link6_cyl) for link6_cyl in link6_cyls]
collision_pts_pos.update(dict(zip(
    map(lambda node: node.name, link6_cyls_pos),
    link6_cyls_pos)))


# add collision avoidance control points to fingers
fing1_base_proj = ProjectionNode("fing1_base_proj", root, np.array([1] * 7 + [0] * 5))
fing1_base_cyls = kdl_cylinder("fing1_base_cyl", fing1_base_proj, robot, 'world', 'j2s6s300_link_finger_1',
                               r=0.01, h=0.025, pts_in_h=2, pts_per_round=3, link_dir=np.array([1,0,0]).reshape(-1,1))
fing1_base_cyls_pos = [PositionProjection(fing1_base_cyl.name + "_pos", fing1_base_cyl) for fing1_base_cyl in fing1_base_cyls]
collision_pts_pos.update(dict(zip(
    map(lambda node: node.name, fing1_base_cyls_pos),
    fing1_base_cyls_pos)))

fing1_tip_proj = ProjectionNode("fing1_tip_proj", root, np.array([1] * 8 + [0] * 4))
fing1_tip_cyls = kdl_cylinder("fing1_tip_cyl", fing1_tip_proj, robot, 'world', 'j2s6s300_link_finger_tip_1',
                               r=0.01, h=0.04, pts_in_h=2, pts_per_round=3, link_dir=np.array([1,0,0]).reshape(-1,1))
fing1_tip_cyls_pos = [PositionProjection(fing1_tip_cyl.name + "_pos", fing1_tip_cyl) for fing1_tip_cyl in fing1_tip_cyls]
collision_pts_pos.update(dict(zip(
    map(lambda node: node.name, fing1_tip_cyls_pos),
    fing1_tip_cyls_pos)))



fing2_base_proj = ProjectionNode("fing2_base_proj", root, np.array([1] * 6 + [0] * 2 + [1] + [0] * 3))
fing2_base_cyls = kdl_cylinder("fing2_base_cyl", fing2_base_proj, robot, 'world', 'j2s6s300_link_finger_2',
                               r=0.01, h=0.025, pts_in_h=2, pts_per_round=3, link_dir=np.array([1,0,0]).reshape(-1,1))
fing2_base_cyls_pos = [PositionProjection(fing2_base_cyl.name + "_pos", fing2_base_cyl) for fing2_base_cyl in fing2_base_cyls]
collision_pts_pos.update(dict(zip(
    map(lambda node: node.name, fing2_base_cyls_pos),
    fing2_base_cyls_pos)))

fing2_tip_proj = ProjectionNode("fing2_tip_proj", root, np.array([1] * 6 + [0] * 2 + [1] * 2 + [0] * 2))
fing2_tip_cyls = kdl_cylinder("fing2_tip_cyl", fing2_tip_proj, robot, 'world', 'j2s6s300_link_finger_tip_2',
                              r=0.01, h=0.04, pts_in_h=2, pts_per_round=3, link_dir=np.array([1,0,0]).reshape(-1,1))
fing2_tip_cyls_pos = [PositionProjection(fing2_tip_cyl.name + "_pos", fing2_tip_cyl) for fing2_tip_cyl in fing2_tip_cyls]
collision_pts_pos.update(dict(zip(
    map(lambda node: node.name, fing2_tip_cyls_pos),
    fing2_tip_cyls_pos)))


fing3_base_proj = ProjectionNode("fing3_base_proj", root, np.array([1] * 6 + [0] * 4 + [1] + [0]))
fing3_base_cyls = kdl_cylinder("fing3_base_cyl", fing3_base_proj, robot, 'world', 'j2s6s300_link_finger_3',
                               r=0.01, h=0.025, pts_in_h=2, pts_per_round=3, link_dir=np.array([1,0,0]).reshape(-1,1))
fing3_base_cyls_pos = [PositionProjection(fing3_base_cyl.name + "_pos", fing3_base_cyl) for fing3_base_cyl in fing3_base_cyls]
collision_pts_pos.update(dict(zip(
    map(lambda node: node.name, fing3_base_cyls_pos),
    fing3_base_cyls_pos)))

fing3_tip_proj = ProjectionNode("fing3_tip_proj", root, np.array([1] * 6 + [0] * 4 + [1] * 2))
fing3_tip_cyls = kdl_cylinder("fing3_tip_cyl", fing2_tip_proj, robot, 'world', 'j2s6s300_link_finger_tip_3',
                              r=0.01, h=0.04, pts_in_h=2, pts_per_round=3, link_dir=np.array([1,0,0]).reshape(-1,1))
fing3_tip_cyls_pos = [PositionProjection(fing3_tip_cyl.name + "_pos", fing3_tip_cyl) for fing3_tip_cyl in fing3_tip_cyls]
collision_pts_pos.update(dict(zip(
    map(lambda node: node.name, fing3_tip_cyls_pos),
    fing3_tip_cyls_pos)))



# attach collision avoidance nodes (avoid the frame)
for name, node in list(collision_pts_pos.items()):
    leaves.CollisionAvoidanceBox(name + "_right_avoider", node, None,
                                 np.array([fright_pos]).T, np.array([[0.05, 0.05, 0.10]]).T,
                                 0.005, epsilon=0.0, eta=2, r_w = 0.2, alpha=1e-30)

    leaves.CollisionAvoidanceBox(name + "_left_avoider", node, None,
                                 np.array([fleft_pos]).T, np.array([[0.05, 0.05, 0.10]]).T,
                                 0.005, epsilon=0.0, eta=2, r_w = 0.2, alpha=1e-7)

    leaves.CollisionAvoidanceBox(name + "_bot_avoider", node, None,
                                 np.array([fbot_pos]).T, np.array([[0.1, 0.05, 0.05]]).T,
                                 0.005, epsilon=0.0, eta=2, r_w = 0.2, alpha=1e-7)

    leaves.CollisionAvoidanceBox(name + "_top_avoider", node, None,
                                 np.array([ftop_pos]).T, np.array([[0.1, 0.05, 0.05]]).T,
                                 0.005, epsilon=0.0, eta=2, r_w=0.2, alpha=1e-7)

# attract palm of hand to target
atrc_pos = leaves.GoalAttractorUni("jaco_attractor_pos", link6_exts_pos[1], np.array([target_pos]).T, gain=20)
# atrc_rotx = leaves.GoalAttractorUni("jaco_attractor_rotx", link6_ext_rotx, np.array([[0.0]]), gain=20, alpha=2)
# atrc_roty = leaves.GoalAttractorUni("jaco_attractor_roty", link6_ext_roty, np.array([[0.0]]), gain=20, alpha=2)
# atrc_rotz = leaves.GoalAttractorUni("jaco_attractor_rotz", link6_ext_rotz, np.array([[0.0]]), gain=20, alpha=2)

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
        env.step(action)
    except:
        print("bad qdd: " + str(qdd))
        break

    # print end-effector position and orientation
    print('xpos: ' + str(env.sim.data.body_xpos[7]))
    print('qpos: ' + str(env.sim.data.qpos))
    quat = mjc.quat_to_scipy(env.sim.data.body_xquat[6])
    r = R.from_quat(quat)
    print('rot: ' + str(r.as_euler('xyz', degrees=False)))

    # update site position marking repulsion points
    for i in range(len(list(collision_pts_pos.values()))):
        env.model.site_pos[i] = list(collision_pts_pos.values())[i].x.flatten()
