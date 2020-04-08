import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco_py

def gravity_comp(sim, ndof=9):
    # % qfrc_bias represents sum of Coriolis and gravity forces.
    return sim.data.qfrc_bias[:ndof]

def get_mass_matrix(sim, ndof):

    # % prepare array to hold result
    m = np.ndarray(shape=(len(sim.data.qvel)**2,),
                             dtype=np.float64,
                             order='C')

    # % call mujoco internal inertia matrix fuction
    mujoco_py.cymj._mj_fullM(sim.model, m, sim.data.qM)

    # % reshape to square, slice, and return
    m=m.reshape(len(sim.data.qvel),-1)
    return m[:ndof,:ndof]

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
    qdd = [0]*len(sim.data.qpos[:ndof]) if qdd is None else qdd
    kp = np.eye(len(sim.data.qpos[:ndof]))*10 if kp is None else kp
    kv = np.eye(len(sim.data.qpos[:ndof])) if kv is None else kv

    # % compute the control as above
    m = get_mass_matrix(sim, ndof)
    bias = sim.data.qfrc_bias[:ndof]
    e = q - sim.data.qpos[:ndof]
    ed = qd - sim.data.qvel[:ndof]
    tau_prime = qdd + np.matmul(kp, e) + np.matmul(kv, ed)
    return np.matmul(m, tau_prime) + bias

def jac(sim, body, ndof):
    """
    Computes Jacobian of body using q = sim.data.qpos, qdot = sim.data.qvel.
    returns jacp, jacr (position, orientation jacobians)

    note: I think I will soon concatenate jacp, jacr
    """
    jacp = np.ndarray(shape=(3*len(sim.data.qpos)),
                      dtype=np.float64,
                      order='C')

    jacr = np.ndarray(shape=jacp.shape,
                      dtype=np.float64,
                      order='C')

    mujoco_py.cymj._mj_jacBody(sim.model,
                               sim.data,
                               jacp,
                               jacr,
                               body)
    jacp=jacp.reshape(3,-1)
    jacr=jacr.reshape(3,-1)
    return jacp[:,:ndof], jacr[:,:ndof]

def mj_ik_traj(y_star, T, env, ee_index, ndof=9):
    """
    moves end effector in a straight line in cartesian space from present pose to y_star
    TODO: position only at present - add orientation
    TODO: collision checking
    TODO: use fake environment for planning without execution
    """
    sim=env.sim
    y0 = np.array(sim.data.body_xpos[ee_index])
    q = sim.data.qpos[:ndof]
    qs=[]
    qs.append(q)
    ys = []

    for t in range(1,T):
        y=np.array(sim.data.body_xpos[ee_index])
        q = sim.data.qpos[:ndof]
        qvel=sim.data.qvel[:ndof]

        jacp,jacr = jac(sim, ee_index, ndof)
        y_hat = y0 + ( t*1.0 / (T*1.0) ) * (y_star-y0)

        jacp=jacp.reshape(3,-1)

        # % new joint positions
        q_update = np.linalg.pinv(jacp).dot( (y_hat - y).reshape(3,1) )
        q = q + q_update[:len(sim.data.qpos[:ndof])].reshape(len(sim.data.qpos[:ndof]),)
        qs.append(q)
        ys.append(y)
        action=pd(None,None, qs[t], env.sim)
        env.step(action)

    return qs, ys

def ee_regulation(x_des, sim, ee_index, kp=None, kv=None, ndof=9):
    """
    This is pointless at present, but it is a building block
    for more complex cartesian control.

    PD control with gravity compensation in cartesian space
    returns J^T(q)[kp(x_des - x) - kv(xdot)] + H(q,qdot)

    TODO: quaternions or axis angles for full ee pose.
    """
    kp = np.eye(len(sim.data.body_xpos[ee_index]))*10 if kp is None else kp
    kv = np.eye(len(sim.data.body_xpos[ee_index]))*1 if kv is None else kv

    jacp,jacr=jac(sim, ee_index, ndof)

    # % compute
    xdot = np.matmul(jacp, sim.data.qvel[:ndof])
    error_vel = xdot
    error_pos = x_des - sim.data.body_xpos[ee_index]
    pos_term = np.matmul(kp,error_pos)
    vel_term = np.matmul(kv,error_vel)

    # % commanding ee pose only
    F = pos_term - vel_term
    torques = np.matmul(jacp.T, F) + sim.data.qfrc_bias[:ndof]
    # torques = np.matmul(jacp.T, F)
    return torques

def generate_random_goal(n=9):
    return np.random.rand(n)*np.pi / 2.0

def quat_to_scipy(q):
    """ scalar last, [x,y,z,w]"""
    return [q[1], q[2], q[3], q[0]]

def quat_to_mj(q):
    """ scalar first, [w,x,y,z]"""
    return [q[-1], q[0], q[1], q[2]]

def transform_jacobian(jac, body, sim):
    """
    converts jacp (or jacp_dot) in global frame to body frame
    jacp: np.ndarray (3,n)
    body: integer
    sim: MjSim
    returns: R_0^T J

    TODO: rotation component of Jacobian, too.
    """
    x = sim.data.body_xpos[body]
    quat = sim.data.body_xquat[body]
    r = R.from_quat(quat_to_scipy(quat))
    return np.matmul(r.as_matrix().T, jac)

def compute_jdot(jac, body, sim):
    """
    compute time derivative of jac of body (in global frame)
    assumes: all joints are revolute
    jac: np.ndarray (3,n)
    body: integer
    sim: MjSim
    returns: jdot

    TODO: rotation component of Jacobian, too.
    TODO: finish this
    """
    Jdot_pos = []
    Jdot_rot = []
    for i in range(len(sim.data.qpos)):
        w_prev = -1 # omega_i=qdot_i * zi−1
        z_i = -1# rotation axis in global frame
        Jdot_pos_i = np.cross(w_prev, z_i)
