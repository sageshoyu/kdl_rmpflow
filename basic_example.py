import copy
from scipy.spatial.transform import Rotation as R
import motor_skills.core.mj_control as mjc
from motor_skills.envs.mj_jaco import MjJacoEnv
# %%
env = MjJacoEnv(vis=True)
while True:

    # % compensate for gravity
    # action = mjc.gravity_comp(env.sim)
    # env.step(action)

    # % PD control in start state.
    # q = [0]*9
    # qd = [0]*9
    # qdd = [0]*9
    # action = mjc.pd(qdd,qd,q,env.sim)
    # env.step(action)

    # % PD constant velocity rotation
    env.sim.data.qpos[:6] = [1, 1.5, 2.3, 0.2, 1.1, 1.1]
    qdd=None
    q=[1]*6
    qd=[0]*6
    action = mjc.pd(qdd,qd,q,env.sim,ndof=6)
    env.step(action)
    print('qpos: ' + str(env.sim.data.qpos[:6]))
    print('xpos: ' + str(env.sim.data.body_xpos[6]))
    quat = mjc.quat_to_scipy(env.sim.data.body_xquat[6])
    r = R.from_quat(quat)
    print('rot: ' + str(r.as_euler('xyz', degrees=False)))

    # for q = [1]*6
    # end effector xpos: [ 0.17810713 -0.29558954  0.36501977]
    # KDL-calculated xpos: [0.14183, -0.37499, 0.30897]


    # for q = [1, 1.5, 2.3, 0.2, 1.1, 1.1]
    # end effector xpos: [ 0.33328221 -0.53719429  0.46320837]
    # kdl-calculated xpos: [0.33370, -0.57184, 0.36541]

    # end effector rot: [-0.34034454 - 0.00997718 - 0.01620031]
    # kdl-calculated rot: [-0.34034, -0.00998, -0.1620]

    # TODO: may be close enough - ask Ben if these differences
    # are significant
