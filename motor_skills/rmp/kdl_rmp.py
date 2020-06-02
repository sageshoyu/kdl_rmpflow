from motor_skills.rmp.rmp import RMPNode
from urdf_parser_py.urdf import URDF as u_parser
from kdl_parser_py import urdf as k_parser
import numpy as np
import PyKDL as kdl


class KDLFlatRMPNode(RMPNode):
    def __init__(self, name, parent, urdf_path, base_link, end_link):
        # now we construct the end-effector node from urdf
        # load URDF
        robot = u_parser.from_xml_file(urdf_path)
        _, tree = k_parser.treeFromUrdfModel(robot)
        self.chain = tree.getChain(base_link, end_link)

        # define kinematics solvers
        self.pos_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.jacd_solver = kdl.ChainJntToJacDotSolver(self.chain)

        # forward kinematics
        def psi(q):
            p_frame = kdl.Frame()
            jnt_q = np_to_jnt_arr(q)
            self.pos_solver.JntToCart(jnt_q, p_frame)
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

        super().__init__(name, parent, psi, J, J_dot)


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
