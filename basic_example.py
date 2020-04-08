import copy
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
    qdd=None
    q=None
    qd=[0]*9
    qd[0] += 1.0
    action = mjc.pd(qdd,qd,q,env.sim)
    env.step(action)
