# Leaf node RMP classes
# @author Anqi Li
# @date April 8, 2019

from .rmp import RMPNode, RMPRoot, RMPLeaf
import numpy as np
from numpy.linalg import norm

import jax.numpy as jnp
from jax import grad, jit, vmap


class CollisionAvoidanceGeorgia(RMPLeaf):
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
                w = 1.0 / x ** 4
                grad_w = -4.0 / x ** 5

            # epsilon is the constant value when moving away from the obstacle
            u = epsilon + np.minimum(0, x_dot) * x_dot
            g = w * u

            grad_u = 2 * np.minimum(0, x_dot)
            grad_Phi = alpha * w * grad_w
            xi = 0.5 * x_dot ** 2 * u * grad_w

            # upper-case xi calculation is included here
            M = g + 0.5 * x_dot * w * grad_u
            M = np.minimum(np.maximum(M, - 1e5), 1e5)

            Bx_dot = eta * g * x_dot

            f = - grad_Phi - xi - Bx_dot
            # remember: this is modified a TON
            f = np.minimum(np.maximum(f, - 1e10), 1e10)

            # print(self.name + " f: " + str(f))
            # print(self.name + " M: " + str(M))
            # print(self.name + " g: " + str(g))

            return (f, M)

        RMPLeaf.__init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func)


class CollisionAvoidance(RMPLeaf):
    """
    Obstacle avoidance RMP leaf, but with recommendations from the original RMPFlow
    paper
    """

    def __init__(self, name, parent, parent_param, c, R=1, epsilon=0.2,
                 alpha=1e-5, eta=0, r_w=0.07, sigma=0.5):

        self.R = R
        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon
        self.r_w = r_w

        if parent_param:
            psi = None
            J = None
            J_dot = None

        else:
            if c.ndim == 1:
                c = c.reshape(-1, 1)

            N = c.size
            # R is the radius of the obstacle point
            psi = lambda y: np.array(norm(y - c) - R).reshape(-1, 1)
            J = lambda y: 1.0 / norm(y - c) * (y - c).T
            J_dot = lambda y, y_dot: np.dot(
                y_dot.T,
                (-1 / norm(y - c) ** 3 * np.dot((y - c), (y - c).T)
                 + 1 / norm(y - c) * np.eye(N)))

        self.w = lambda y: max(self.r_w - y, 0) / (y - R) if y >= 0 else 1e10
        self.grad_w = grad(self.w)

        self.u = lambda y_dot: epsilon + (1.0 - jnp.exp(-y_dot ** 2 / 2.0 / sigma ** 2) if y_dot < 0 else 0.0)
        self.grad_u = grad(self.u)

        # computations done in Jax and returne din numpy
        def RMP_func(x, x_dot):

            x = x[0][0]
            x_dot = x_dot[0][0]

            w_x = self.w(x)
            dw_x = self.grad_w(x)

            # epsilon is the constant value when moving away from the obstacle
            u_xd = self.u(x_dot)

            g = w_x * u_xd

            du_xd = self.grad_u(x_dot)

            grad_Phi = alpha * w_x * dw_x
            xi = 0.5 * x_dot ** 2 * u_xd * dw_x

            # upper-case xi calculation is included here
            M = g + 0.5 * x_dot * w_x * du_xd
            M = jnp.minimum(jnp.maximum(M, - 1e5), 1e5)

            Bx_dot = eta * g * x_dot

            f = - grad_Phi - xi - Bx_dot
            # remember: this is modified a TON
            f = jnp.minimum(jnp.maximum(f, - 1e10), 1e10)

            # print(self.name + " f: " + str(f))
            # print(self.name + " M: " + str(M))
            # print(self.name + " g: " + str(g))

            # convert from jax array to numpy array and return
            return (np.asarray(f), np.asarray(M))

        RMPLeaf.__init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func)


# todo: policy currently unstable - need to check math (use auto-differentiation library like Jax?)
class CollisionAvoidanceBox(RMPLeaf):
    """
    Obstacle avoidance RMP leaf
    """

    def __init__(self, name, parent, parent_param, c, r, epsilon=0.2,
                 alpha=1e-5, eta=0):
        r = np.abs(r)
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
            # graphics people solved this one already:
            # https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
            # note: we normalize by R

            # TODO: implement length, width, height normalization
            psi = lambda y: np.array(
                (norm(np.maximum(np.abs(y - c) - r, 0.0))
                 + min(np.max(np.abs(y - c) - r), 0.0))).reshape(-1, 1)

            # (but not the Jacobian)
            def J(y):
                p_min_r = np.maximum((np.abs(y - c) - r), 0.0)
                norm_p_min_r = np.array(norm(p_min_r), dtype=float).reshape(-1, 1)
                p_out = np.divide(p_min_r, norm_p_min_r, out=np.zeros_like(p_min_r), where=norm_p_min_r != 0)
                p_in = np.zeros((np.size(c), 1))
                p_in[np.argmax(p_min_r)][0] = -int(np.all(np.less(p_min_r, 0)))
                return ((p_out + p_in) * np.sign(y - c)).T

            self.J = J

            # ... and J dot
            def J_dot(y, y_dot):
                p_min_r = np.maximum((np.abs(y - c) - r), 0.0)
                p_min_r3 = p_min_r ** 3
                norm_p_min_r = np.array(norm(p_min_r), dtype=float).reshape(-1, 1)
                fp_g = np.divide(y_dot, norm_p_min_r, out=np.zeros_like(y_dot), where=norm_p_min_r != 0)
                p_fg = np.dot(p_min_r.T, y_dot) \
                       * np.divide(p_min_r, p_min_r3, out=np.zeros_like(p_min_r), where=p_min_r3 != 0)
                return ((fp_g - p_fg) * np.sign(y - c)).T

            self.J_dot = J_dot

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

            # print(self.name + " f: " + str(f))
            # print(self.name + " M: " + str(M))
            # print(self.name + " g: " + str(g))

            return (f, M)

        RMPLeaf.__init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func)


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

            # gamma(x) in RMPFlow paper
            beta = np.exp(- x_norm ** 2 / 2 / (sigma ** 2))
            w = (w_u - w_l) * beta + w_l
            s = (1 - np.exp(-2 * alpha * x_norm)) / (1 + np.exp(
                -2 * alpha * x_norm))

            # this is potential suggested in RMPFlow appendix
            G = np.eye(N) * w
            if x_norm > tol:
                grad_Phi = s / x_norm * w * x * gain
            else:
                grad_Phi = 0
            Bx_dot = eta * w * x_dot
            grad_w = - beta * (w_u - w_l) / sigma ** 2 * x

            # since gradient is simple, we xi is hand-computed,
            # M_stretch is a bit more complicated, we'll be using
            # a differentiation library like AutoGrad
            x_dot_norm = norm(x_dot)
            xi = -0.5 * (x_dot_norm ** 2 * grad_w - 2 *
                         np.dot(np.dot(x_dot, x_dot.T), grad_w))

            # no dependence on velocity, so upper-case XI = 0
            M = G
            f = - grad_Phi - Bx_dot - xi


            return (f, M)

        RMPLeaf.__init__(self, name, parent, None, psi, J, J_dot, RMP_func)

    # TODO: implement M_stretch for more obstacle-heavy environments

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
        J = lambda y: np.eye(y.size)
        J_dot = lambda y, y_dot: np.zeros((y.size, y.size))

        def RMP_func(x, x_dot):
            G = w
            Bx_dot = eta * w * x_dot
            M = G
            f = - Bx_dot

            return (f, M)

        RMPLeaf.__init__(self, name, parent, None, psi, J, J_dot, RMP_func)


class JointLimiter(RMPLeaf):
    def __init__(self, name, parent, jnt_bounds, x_0, lam=0.01, sigma=1, nu_p=1e-5, nu_d=1e-5):
        psi = lambda y: y
        J = lambda y: np.eye(y.size)
        J_dot = lambda y, y_dot: np.zeros((y.size, y.size))

        l_l = jnp.asarray(jnt_bounds[:, [0]])
        l_u = jnp.asarray(jnt_bounds[:, [1]])

        def d(q, qd, ll_l, ll_u):
            s = (q - ll_l) / (ll_u - ll_l)
            d = 4 * s * (1 - s)
            alpha_l = 1 - jnp.exp(-jnp.minimum(qd, 0) ** 2 / 2 / sigma ** 2)
            alpha_u = 1 - jnp.exp(-jnp.maximum(qd, 0) ** 2 / 2 / sigma ** 2)
            return (s * (alpha_u * d + (1 - alpha_u)) + (1 - s) * (alpha_l * d + (1 - alpha_l))) ** -2


        grad_d = vmap(grad(d, argnums=0), in_axes=(0, 0, 0, 0), out_axes=0)

        def RMP_func(x, x_dot):
            x = jnp.asarray(x)
            x_dot = jnp.asarray(x_dot)

            M = lam * jnp.diag(d(x, x_dot, l_l, l_u).flatten())
            # must flatten since jax can only evaluate scalars
            xi = (0.5 * grad_d(x.flatten(), x_dot.flatten(), l_l.flatten(), l_u.flatten()).reshape(-1, 1) * x_dot ** 2)

            f = jnp.dot(M, nu_p * (x_0 - x) - nu_d * x_dot) - xi

            # print(self.name + " f: " + str(f))
            # print(self.name + " M: " + str(M))

            return (np.asarray(f), np.asarray(M))

        super().__init__(name, parent, None, psi, J, J_dot, RMP_func)
