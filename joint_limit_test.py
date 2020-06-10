from motor_skills.rmp.rmp import RMPRoot
from motor_skills.rmp.rmp_leaf import JointLimiter
import numpy as np

bounds = np.array([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])

root = RMPRoot("root")
limit = JointLimiter("limiter", root, bounds, np.zeros((4,1)), lam=1)

# at 0 position
print("resting: " + str(root.solve(np.zeros((4,1)), np.zeros((4,1)))))

# at lower limit, heading towards lower limit
ql = np.array([-0.999, -0.999, -0.999, -0.999]).reshape(-1, 1)
vl = np.array([-2.5, -2.5, -2.5, -2.5]).reshape(-1, 1)
print("lower trigger: " + str(root.solve(ql, vl)))

# at lower limit, heading away from lower limit
vu = np.array([2.5, 2.5, 2.5, 2.5]).reshape(-1, 1)
print("lower nontrigger: " + str(root.solve(ql, vu)))

# at upper limit, heading towards upper limit
qu = np.array([0.999, 0.999, 0.999, 0.999]).reshape(-1, 1)
print("upper trigger: " + str(root.solve(qu, vu)))

# at upper limit, heaving away from upper limit
print("upper nontrigger: " + str(root.solve(qu, vl)))