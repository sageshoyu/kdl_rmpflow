# Leaf node RMP classes
# @author Anqi Li
# @date April 8, 2019

from .rmp import RMPNode, RMPRoot, RMPLeaf
import numpy as np
from numpy.linalg import norm, inv
from scipy.spatial.transform import Rotation as Rot
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd


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

# TODO: generalize this class and just give distance funcs + Jacobians (RMPFunc is
# repeated three times at this point)
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

        def RMP_func(x, x_dot):
            w = max(r_w - x, 0) / (x - R) if x >= 0 else 1e10
            grad_w = (((r_w - x) > 0) * -1 * (x - R) - max(r_w - x, 0.0)) / (x - R) ** 2 \
                if x >= 0 else 0

            # epsilon is the constant value when moving away from the obstacle
            u = epsilon + (1.0 - np.exp(-x_dot ** 2 / 2.0 / sigma ** 2) if x_dot < 0 else 0.0)
            g = w * u

            grad_u = np.exp(-x_dot ** 2 / 2.0 / sigma ** 2) * x_dot / sigma ** 2 if x_dot < 0 else 0.0

            grad_Phi = alpha * w * grad_w
            xi = 0.5 * x_dot ** 2 * u * grad_w

            # upper-case xi calculation is included here
            M = g + 0.5 * x_dot * w * grad_u
            M = np.minimum(np.maximum(M, - 1e5), 1e5)

            Bx_dot = eta * g * x_dot

            f = - grad_Phi - xi - Bx_dot
            # remember: this is modified a TON
            f = np.minimum(np.maximum(f, - 1e10), 1e10)

            # convert from jax array to numpy array and return
            return (f, M)

        RMPLeaf.__init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func)

class CollisionAvoidanceBox(RMPLeaf):
    """
    Obstacle avoidance RMP leaf
    """
    def __init__(self, name, parent, parent_param, c, r, R,
                 xyz=np.zeros((3, 1)), epsilon=0.2, alpha=1e-5, eta=0, r_w=0.07, sigma=0.5):
        r = np.abs(r)
        rot = Rot.from_euler('xyz', xyz.flatten())

        # rotation transformation from global frame to box
        rot_inv = inv(rot.as_matrix())
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

            # graphics people solved this one already:
            # https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
            # note: we normalize by R

            def psi(y):
                q = np.abs(np.dot(rot_inv, y - c)) - r
                return np.array(norm(np.maximum(q, 0.0))).reshape(-1, 1)

            # (but not the Jacobian)
            # leveraged the Jacobian flavor of chain rule here
            def J(y):
                p = np.dot(rot_inv, y - c)
                q = np.abs(p) - r
                sdf = norm(np.maximum(q, 0.0))
                if sdf == 0:
                    print("WARNING: BOX SDF IS ZERO")

                return np.dot(np.repeat(1 / sdf, 3).reshape(1, 3),
                       np.dot(np.diag(np.maximum(q, 0.0).flatten()),
                              np.dot(np.diag(np.sign(p).flatten()),
                                     rot_inv)))

            # ... and J dot (this was done by multiplying out Jacobian and using quotient rule)
            def J_dot(y, y_dot):
                p = np.dot(rot_inv, y - c)
                q = np.abs(p) - r
                sdf = norm(np.maximum(q, 0.0))

                p_dot = np.dot(rot_inv, y_dot)
                q_dot = np.sign(p) * p_dot
                max_dot = (q > 0) * q_dot

                sdf_dot = np.sum(np.maximum(q, 0.0) * max_dot) / sdf
                return np.dot(
                    (np.sign(p) * (max_dot * sdf - sdf_dot * np.maximum(q, 0.0)) / sdf ** 2).T,
                    rot_inv)

        def RMP_func(x, x_dot):
            w = max(r_w - x, 0) / (x - R) if (x - R) >= 0 else 1e10
            grad_w = (((r_w - x) > 0) * -1 * (x - R) - max(r_w - x, 0.0)) / (x - R) ** 2 \
                if (x - R) >= 0 else 0

            # epsilon is the constant value when moving away from the obstacle
            u = epsilon + (1.0 - np.exp(-x_dot ** 2 / 2.0 / sigma ** 2) if x_dot < 0 else 0.0)
            g = w * u

            grad_u = np.exp(-x_dot ** 2 / 2.0 / sigma ** 2) * x_dot / sigma ** 2 if x_dot < 0 else 0.0

            grad_Phi = alpha * w * grad_w
            xi = 0.5 * x_dot ** 2 * u * grad_w

            # upper-case xi calculation is included here
            M = g + 0.5 * x_dot * w * grad_u
            M = np.minimum(np.maximum(M, - 1e5), 1e5)

            Bx_dot = eta * g * x_dot

            f = - grad_Phi - xi - Bx_dot
            f = np.minimum(np.maximum(f, - 1e10), 1e10)

            return (f, M)

        RMPLeaf.__init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func)


class CollisionAvoidancePlane(RMPLeaf):
    """
    Obstacle avoidance RMP leaf for infinite plane (orientation-sensitive)
    """
    def __init__(self, name, parent, parent_param, c, R, n=np.array([[0, 0, 1]]).T, epsilon=0.2, alpha=1e-5, eta=0, r_w=0.07, sigma=0.5):

        # normalize normal in case it hasn't already
        n = n / norm(n)
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

            def psi(y):
                return np.dot(n.T, y - c) - R

            # (but not the Jacobian)
            # leveraged the Jacobian flavor of chain rule here
            def J(y):
                return n.T
            # ... and J dot (this was done by multiplying out
            def J_dot(y, y_dot):
                return np.zeros((1,3))

        def RMP_func(x, x_dot):
            w = max(r_w - x, 0) / (x - R) if x >= 0 else 1e10
            grad_w = (((r_w - x) > 0) * -1 * (x - R) - max(r_w - x, 0.0)) / (x - R) ** 2 \
                if x >= 0 else 0

            # epsilon is the constant value when moving away from the obstacle
            u = epsilon + (1.0 - np.exp(-x_dot ** 2 / 2.0 / sigma ** 2) if x_dot < 0 else 0.0)
            g = w * u

            grad_u = np.exp(-x_dot ** 2 / 2.0 / sigma ** 2) * x_dot / sigma ** 2 if x_dot < 0 else 0.0

            grad_Phi = alpha * w * grad_w
            xi = 0.5 * x_dot ** 2 * u * grad_w

            # upper-case xi calculation is included here
            M = g + 0.5 * x_dot * w * grad_u
            M = np.minimum(np.maximum(M, - 1e5), 1e5)

            Bx_dot = eta * g * x_dot

            f = - grad_Phi - xi - Bx_dot
            # remember: this is modified a TON
            f = np.minimum(np.maximum(f, - 1e10), 1e10)

            # convert from jax array to numpy array and return
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
    def __init__(self, name, parent, jnt_bounds, x_0, lam=0.01, sigma=0.1, nu_p=1e-5, nu_d=1e-5):
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
