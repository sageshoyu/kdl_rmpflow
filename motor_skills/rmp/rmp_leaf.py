# Leaf node RMP classes
# @author Anqi Li
# @date April 8, 2019

from .rmp import RMPNode, RMPRoot, RMPLeaf
import numpy as np
from numpy.linalg import norm


class CollisionAvoidance(RMPLeaf):
    """
    Obstacle avoidance RMP leaf
    """

    def __init__(self, name, parent, parent_param, c, R=1, epsilon=0.2,
                 alpha=1e-5, eta=0):

        self.R = R
        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon

        if parent_param:
            psi = None
            J = None
            J_dot = None

        else:
            if c.ndim == 1:
                c = c.reshape(-1, 1)

            N = c.size
            # R is the radius of the obstacle point
            psi = lambda y: np.array(norm(y - c) / R - 1).reshape(-1, 1)
            J = lambda y: 1.0 / norm(y - c) * (y - c).T / R
            J_dot = lambda y, y_dot: np.dot(
                y_dot.T,
                (-1 / norm(y - c) ** 3 * np.dot((y - c), (y - c).T)
                 + 1 / norm(y - c) * np.eye(N))) / R

        def RMP_func(x, x_dot):
            # if inside obstacle, set w to HIGH value to PULL OUT
            if x < 0:
                w = 1e10
                grad_w = 0
            # if not, decrease pressure according to power of 2 (previously pwr of 4, too aggressive)
            else:
                w = 1.0 / x ** 2
                grad_w = -2.0 / x ** 3
            # epsilon is the constant value when moving away from the obstacle
            u = epsilon + np.minimum(0, x_dot) * x_dot
            g = w * u

            grad_u = 2 * np.minimum(0, x_dot)
            grad_Phi = alpha * w * grad_w
            xi = 0.5 * x_dot ** 2 * u * grad_w

            M = g + 0.5 * x_dot * w * grad_u
            M = np.minimum(np.maximum(M, - 1e5), 1e5)

            Bx_dot = eta * g * x_dot

            f = - grad_Phi - xi - Bx_dot
            # remember: this is modified a TON
            f = np.minimum(np.maximum(f, - 1e10), 1e10)

            #print(self.name + " f: " + str(f))
            #print(self.name + " M: " + str(M))
            #print(self.name + " g: " + str(g))

            return (f, M)

        RMPLeaf.__init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func)

    def set_obstacle(self, c):
        self.psi = lambda y: np.array(norm(y - c) / self.R - 1).reshape(-1, 1)


class GoalAttractorUni(RMPLeaf):
    """
    Goal Attractor RMP leaf
    """

    def __init__(self, name, parent, y_g, w_u=10, w_l=1, sigma=1,
                 alpha=1, eta=2, gain=1, tol=0.005):

        if y_g.ndim == 1:
            y_g = y_g.reshape(-1, 1)
        N = y_g.size
        psi = lambda y: (y - y_g)
        J = lambda y: np.eye(N)
        J_dot = lambda y, y_dot: np.zeros((N, N))

        def RMP_func(x, x_dot):
            x_norm = norm(x)

            beta = np.exp(- x_norm ** 2 / 2 / (sigma ** 2))
            w = (w_u - w_l) * beta + w_l
            s = (1 - np.exp(-2 * alpha * x_norm)) / (1 + np.exp(
                -2 * alpha * x_norm))

            G = np.eye(N) * w
            if x_norm > tol:
                grad_Phi = s / x_norm * w * x * gain
            else:
                grad_Phi = 0
            Bx_dot = eta * w * x_dot
            grad_w = - beta * (w_u - w_l) / sigma ** 2 * x

            x_dot_norm = norm(x_dot)
            xi = -0.5 * (x_dot_norm ** 2 * grad_w - 2 *
                         np.dot(np.dot(x_dot, x_dot.T), grad_w))

            M = G
            f = - grad_Phi - Bx_dot - xi

            #print(self.name + " f: " + str(f))

            return (f, M)

        RMPLeaf.__init__(self, name, parent, None, psi, J, J_dot, RMP_func)

    def update_goal(self, y_g):
        """
        update the position of the goal
        """

        if y_g.ndim == 1:
            y_g = y_g.reshape(-1, 1)
        N = y_g.size
        self.psi = lambda y: (y - y_g)
        self.J = lambda y: np.eye(N)
        self.J_dot = lambda y, y_dot: np.zeros((N, N))


class Damper(RMPLeaf):
    """
    Damper RMP leaf
    """

    def __init__(self, name, parent, w=1, eta=1):
        psi = lambda y: y
        J = lambda y: np.eye(2)
        J_dot = lambda y, y_dot: np.zeros((2, 2))

        def RMP_func(x, x_dot):
            G = w
            Bx_dot = eta * w * x_dot
            M = G
            f = - Bx_dot

            return (f, M)

        RMPLeaf.__init__(self, name, parent, None, psi, J, J_dot, RMP_func)
