from kdl_rmpflow.rmp.rmp import RMPRoot, RMPNode
from kdl_parser_py import urdf as k_parser
from scipy.spatial.transform import Rotation as R
import numpy as np
import PyKDL as kdl


class KDLRMPNode(RMPNode):
    """
    Builds a new RMP node, map is forward kinematics from base_link to end_link.
    offset is in the local frame of end_link's parent joint.
    """
    def __init__(self, name, parent, robot, base_link, end_link, offset=np.zeros((3, 1))):
        _, tree = k_parser.treeFromUrdfModel(robot)
        self.chain = tree.getChain(base_link, end_link)

        # define kinematics solvers
        self.pos_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.jacd_solver = kdl.ChainJntToJacDotSolver(self.chain)

        # forward kinematics
        def psi(q):
            base = np_to_vect(offset)
            p_frame = kdl.Frame()
            jnt_q = np_to_jnt_arr(q)
            e = self.pos_solver.JntToCart(jnt_q, p_frame)
            if e != 0:
                print("KDL SOLVER ERROR in " + name + ": " + str(e))
            p = p_frame * base
            Rz, Ry, Rx = p_frame.M.GetEulerZYX()
            return np.array([[p.x(), p.y(), p.z(), Rx, Ry, Rz]]).T

        # Jacobian for forward kinematics
        def J(q):
            # set of solver inputs
            base = np_to_vect(offset)
            nq = np.size(q)
            jnt_q = np_to_jnt_arr(q)
            jac = kdl.Jacobian(nq)

            # solve Jacobian and transfer into np array
            self.jac_solver.JntToJac(jnt_q, jac)
            jac.changeRefPoint(base)
            return jac_to_np(jac)

        # Jacobian time-derivative of forward kinematics
        def J_dot(q, qd):
            # set solver inputs
            base = np_to_vect(offset)
            nq = np.size(q)
            jnt_q = np_to_jnt_arr(q)
            jnt_qd = np_to_jnt_arr(qd)
            jnt_q_qd = kdl.JntArrayVel(jnt_q, jnt_qd)
            jacd = kdl.Jacobian(nq)
            jacd.changeRefPoint(base)

            # solve and convert to np array
            self.jacd_solver.JntToJacDot(jnt_q_qd, jacd)
            return jac_to_np(jacd)

        super().__init__(name, parent, psi, J, J_dot, verbose=False)


class ProjectionNode(RMPNode):
    """
    Constructs a new node with map that passes through parameters
    (in the same order) as specified by param_map.
    Param_map is the same length of state vector of parent node,
    with 1's in the indices for parameters to be passed, and 0's in
    the indices for parameters to be withheld.
    """
    def __init__(self, name, parent, param_map):
        # construct matrix map, this is for object creation so performance
        # is less of a concern
        one_map = param_map.astype('int32')
        mat = np.zeros((np.sum(one_map), one_map.size), dtype='float64')
        jacd = np.zeros_like(mat)

        i_mat = 0
        for i in range(0, one_map.size):
            if one_map[i] == 1:
                mat[i_mat][i] = 1
                i_mat += 1

        psi = lambda y: np.dot(mat, y)
        super().__init__(name, parent, psi, lambda x: mat, lambda x, xd: jacd)


class PositionProjection(ProjectionNode):
    """
    Convenience method to pass position from KDLRMPNode state.
    """
    def __init__(self, name, parent):
        super().__init__(name, parent, np.array([1, 1, 1, 0, 0, 0]))


class RotZProjection(ProjectionNode):
    """
    Convenience method to pass z-axis rotation from KDLRMPNode state.
    """
    def __init__(self, name, parent):
        super().__init__(name, parent, np.array([0, 0, 0, 0, 0, 1]))


class RotYProjection(ProjectionNode):
    """
    Convenience method to pass y-axis rotation from KDLRMPNode state.
    """
    def __init__(self, name, parent):
        super().__init__(name, parent, np.array([0, 0, 0, 0, 1, 0]))


class RotXProjection(ProjectionNode):
    """
    Convenience method to pass x-axis rotation from KDLRMPNode state.
    """
    def __init__(self, name, parent):
        super().__init__(name, parent, np.array([0, 0, 0, 1, 0, 0]))


def np_to_vect(v):
    return kdl.Vector(v[0], v[1], v[2])


def np_to_jnt_arr(arr):
    nq = np.size(arr)
    jnt_arr = kdl.JntArray(nq)
    for i in range(0, nq):
        jnt_arr[i] = arr[i]

    return jnt_arr


def jac_to_np(jac):
    nq = jac.columns()
    np_jac = np.zeros((6, nq))
    for c in range(0, nq):
        c_twst = jac.getColumn(c)
        for r in range(0, 6):
            np_jac[r][c] = c_twst[r]

    return np_jac


def rmp_from_urdf(robot):
    """
    Constructs rmpflow tree from robot urdf, exposing all actuatable
    joints.

    Returns: root, leaf_dict
    root - root of rmpflow tree
    leaf_dict - dictionary containing all RMPNodes of actuatable
    joints, indexed by joint name as specified in URDF
    """
    # find all actuated joint names
    flatten = lambda l: [item for sublist in l for item in sublist]
    jnts = flatten(list(map(lambda t: t.joints, robot.transmissions)))
    jnt_names = list(map(lambda j: j.name, jnts))

    # find all actuated link names
    # ASSUMPTION: joints are listed the same way as links in both mujoco xml and urdf
    link_names = [robot.joint_map[jnt_name].child for jnt_name in jnt_names]

    # construct RMP bush root
    root = RMPRoot('root')

    # using link names, construct RMP bush
    qlen = len(link_names)
    leaf_dict = {}
    proj_dict = {}

    # construct branch for each segment
    for i in range(1, qlen + 1):
        seg_name = link_names[i - 1]

        # compute which joint angles should be available to node
        # first find parent proj vect
        proj_vect = np.copy(proj_dict.get(robot.parent_map[seg_name][1], np.array([0] * qlen)))

        # index represents actuator joint angle
        proj_vect[i - 1] = 1

        # store in dictionary for child link's reference
        proj_dict[seg_name] = proj_vect

        proj_node = ProjectionNode('proj_' + seg_name, root, proj_vect)
        seg_node = KDLRMPNode(seg_name, proj_node, robot, 'world', seg_name)
        leaf_dict[seg_name] = seg_node

    return root, leaf_dict


def kdl_node_array(name, parent, robot, base_link, end_link, h, num, link_dir, skip_h=0, offset=np.zeros((3,1))):
    """
    Constructs a line of regulary-spaced KDLRMPNodes offset from joint specified by end_link,
    in the unit-direction of link_dir (in local frame of parent joint of end_link).
    """
    assert num >= 2
    unit_dir = link_dir / np.linalg.norm(link_dir)
    spacing = (h - skip_h) / (num - 1)
    nodes = []
    for i in range(num):
        nodes.append(KDLRMPNode(name + str(i),
                                parent,
                                robot,
                                base_link,
                                end_link,
                                offset=unit_dir * spacing * i + offset))
    return nodes


def kdl_cylinder(name,
                 parent,
                 robot,
                 base_link,
                 end_link,
                 r,
                 h,
                 pts_per_round,
                 pts_in_h,
                 link_dir,
                 offset=np.zeros((3,1))):
    """
    Constructs control points in cylindrical formation around arm link
    (typical use case: approximate geometry of link for collision avoidance)

    Keyword Args:
        name -- name prefix for all constructed nodes
        parent -- parent RMPNode (typically root or ProjectionNode from root)
        robot -- robot urdf as constructed by urdf_parser_py
        base_link -- starting link name in URDF
        end_link -- ending link name in URDF
        r -- radius of cylinder
        h -- height/length of cylinder
        pts_per_round -- number of control points in circular cross section of cylinder
        pts_in_h -- number of control points along cylinder length
        link_dir -- unit direction to specify direction of cylinder length
                    (in local frame of parent joint of base_link)
        offset -- length of cylinder to skip (starting from joint) to place first round of control points

    Node: control points are placed AFTER offset is taken into account. So the distance between each round
    of control points is: (h - offset) / pts_in_h
    """

    unit_dir = link_dir / np.linalg.norm(link_dir)

    # np.cross only works with row vecs, so take transpose and go back
    # to find starting vect, take cross with x y and z axis (in that order
    # in case the first one fails)
    rnd_start = np.cross(unit_dir.T, np.array([1,0,0])).T

    if not np.any(rnd_start):
        rnd_start = np.cross(unit_dir.T, np.array([0,1,0])).T

        if not np.any(rnd_start):
            rnd_start = np.cross(unit_dir.T, np.array([0,0,1])).T

    nodes = []

    # position vector from jnt to round pts will be normal to link_dir
    rnd_spacing = 2 * np.pi / pts_per_round

    for i in range(pts_per_round):
        angle_offset = i * rnd_spacing

        # rotate rnd_start about link_dir axis by angle_offset, multiply by radius
        rnd_next = r * R.from_rotvec(angle_offset * unit_dir.T).apply(rnd_start.T).T

        # create node line, offset by rotated rnd_next vector (i.e. revolving line offset by radius)
        nodes.extend(kdl_node_array("round_arr_" + str(i) + "_" + name,
                                    parent,
                                    robot,
                                    base_link,
                                    end_link,
                                    h,
                                    pts_in_h,
                                    unit_dir,
                                    offset=offset + rnd_next))

    return nodes






