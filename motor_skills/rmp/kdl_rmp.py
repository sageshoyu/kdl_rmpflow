from motor_skills.rmp.rmp import RMPRoot, RMPNode
from kdl_parser_py import urdf as k_parser
import numpy as np
import PyKDL as kdl


class KDLRMPNode(RMPNode):
    def __init__(self, name, parent, tree, base_link, end_link):
        self.chain = tree.getChain(base_link, end_link)

        # define kinematics solvers
        self.pos_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.jacd_solver = kdl.ChainJntToJacDotSolver(self.chain)

        # forward kinematics
        def psi(q):
            p_frame = kdl.Frame()
            jnt_q = np_to_jnt_arr(q)
            e = self.pos_solver.JntToCart(jnt_q, p_frame)
            if e != 0:
                print("KDL SOLVER ERROR: " + str(e))
            p = p_frame.p
            return np.array([[p.x(), p.y(), p.z()]]).T

        # Jacobian for forward kinematics
        def J(q):
            # set of solver inputs
            nq = np.size(q)
            jnt_q = np_to_jnt_arr(q)
            jac = kdl.Jacobian(nq)

            # solve Jacobian and transfer into np array
            self.jac_solver.JntToJac(jnt_q, jac)
            return jac_to_np(jac)

        # Jacobian time-derivative of forward kinematics
        def J_dot(q, qd):
            # set solver inputs
            nq = np.size(q)
            jnt_q = np_to_jnt_arr(q)
            jnt_qd = np_to_jnt_arr(qd)
            jnt_q_qd = kdl.JntArrayVel(jnt_q, jnt_qd)
            jacd = kdl.Jacobian(nq)

            # solve and convert to np array
            self.jacd_solver.JntToJacDot(jnt_q_qd, jacd)
            return jac_to_np(jacd)

        super().__init__(name, parent, psi, J, J_dot, verbose=False)


class ProjectionNode(RMPNode):
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


def np_to_jnt_arr(arr):
    nq = np.size(arr)
    jnt_arr = kdl.JntArray(nq)
    for i in range(0, nq):
        jnt_arr[i] = arr[i]

    return jnt_arr


def jac_to_np(jac):
    nq = jac.columns()
    # used to be 6 to include rotation`
    np_jac = np.zeros((3, nq))
    for c in range(0, nq):
        c_twst = jac.getColumn(c)
        # used to be 6 to include rotation
        for r in range(0, 3):
            np_jac[r][c] = c_twst[r]

    return np_jac


# untested
def rmptree_from_chain(tree, chain):
    # find all directly-actuated segments (in order of chain)
    mov_segs = []
    for i in range(0, chain.getNrOfSegments()):
        seg = chain.getSegment(i)
        jnt = seg.getJoint()
        if (jnt.getType() != getattr(kdl.Joint, 'None')):
            mov_segs.append(seg.getName())

    # length of state vect is # of actuated joints
    qlen = len(mov_segs)

    # construct root
    root = RMPRoot('root')

    leaf_dict = {}
    # construct branch for each segment
    for i in range(1, qlen + 1):
        seg_name = mov_segs[i - 1]
        proj_node = ProjectionNode('proj ' + seg_name, root, np.array([1] * i + [0] * (qlen - i)))
        seg_node = KDLRMPNode('name ' + seg_name, proj_node, tree, 'world', seg_name)
        leaf_dict[seg_name] = seg_node

    return root, leaf_dict


def tree_from_robot(robot):
    # find all actuated joint names
    flatten = lambda l: [item for sublist in l for item in sublist]
    jnts = flatten(list(map(lambda t: t.joints, robot.transmissions)))
    jnt_names = list(map(lambda j: j.name, jnts))

    # find all actuated link names
    # ASSUMPTION: joints are listed the same way as links in both mujoco xml and urdf
    link_names = [robot.joint_map[jnt_name].child for jnt_name in jnt_names]

    # construct RMP bush
    root = RMPRoot('root')

    # using link names, construct RMP bush
    qlen = len(link_names)
    leaf_dict = {}
    proj_dict = {}
    _, tree = k_parser.treeFromUrdfModel(robot)

    # construct branch for each segment
    for i in range(1, qlen + 1):
        seg_name = link_names[i - 1]

        # compute which joint angles should be available to node
        # first find parent proj vect
        proj_vect = np.copy(proj_dict.get(robot.parent_map[seg_name][1], np.array([0]*qlen)))

        # index represents actuator joint angle
        proj_vect[i - 1] = 1

        # store in dictionary for child link's reference
        proj_dict[seg_name] = proj_vect

        proj_node = ProjectionNode('proj ' + seg_name, root, proj_vect)
        seg_node = KDLRMPNode(seg_name, proj_node, tree, 'world', seg_name)
        leaf_dict[seg_name] = seg_node

    return root, leaf_dict
