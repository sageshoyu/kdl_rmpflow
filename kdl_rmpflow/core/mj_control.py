import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco_py


def gravity_comp(sim, ndof=9):
    # % qfrc_bias represents sum of Coriolis and gravity forces.
    return sim.data.qfrc_bias[:ndof]


def get_mass_matrix(sim, ndof):
    # % prepare array to hold result
    m = np.ndarray(shape=(len(sim.data.qvel) ** 2,),
                   dtype=np.float64,
                   order='C')

    # % call mujoco internal inertia matrix fuction
    mujoco_py.cymj._mj_fullM(sim.model, m, sim.data.qM)

    # % reshape to square, slice, and return
    m = m.reshape(len(sim.data.qvel), -1)
    return m[:ndof, :ndof]


def pd(qdd, qd, q, sim, kp=None, kv=None, ndof=9):
    """
    inputs (in joint space):
        qdd: desired accel
        qd: desired vel
        q: desire pos
        (if any are None, that term is omitted from the control law)

    kp, kv are scalars in 1D, PSD matrices otherwise
    ndof is the number of degrees of freedom of the robot

    returns M(q)[qdd + kpE + kvEd] + H(q,qdot)
    with E = (q - sim.data.qpos), Ed = (qd - sim.data.qvel), H = sim.data.qfrc_bias
    """

    # % handle None inputs
    q = sim.data.qpos[:ndof] if q is None else q
    qd = sim.data.qvel[:ndof] if qd is None else qd
    qdd = [0] * len(sim.data.qpos[:ndof]) if qdd is None else qdd
    kp = np.eye(len(sim.data.qpos[:ndof])) * 10 if kp is None else kp
    kv = np.eye(len(sim.data.qpos[:ndof])) if kv is None else kv

    # % compute the control as above
    m = get_mass_matrix(sim, ndof)
    bias = sim.data.qfrc_bias[:ndof]
    e = q - sim.data.qpos[:ndof]
    ed = qd - sim.data.qvel[:ndof]
    tau_prime = qdd + np.matmul(kp, e) + np.matmul(kv, ed)
    return np.matmul(m, tau_prime) + bias


def quat_to_scipy(q):
    """ scalar last, [x,y,z,w]"""
    return [q[1], q[2], q[3], q[0]]


def quat_to_mj(q):
    """ scalar first, [w,x,y,z]"""
    return [q[-1], q[0], q[1], q[2]]


