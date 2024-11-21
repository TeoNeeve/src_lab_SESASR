import math
from math import cos, sin, sqrt
import numpy as np
import sympy
from sympy import symbols, Matrix

def squeeze_sympy_out(func):
    # inner function
    def squeeze_out(*args):
        out = func(*args).squeeze()
        return out
    return squeeze_out

def sample_velocity_motion_model_five_state(mu, u, a, dt):
    """ 
    Extended Sample velocity motion model with pose and velocity [x, y, theta, v, w].
    Arguments:
    mu -- state of the robot before moving [x, y, theta, v, w]
    u -- velocity reading obtained from the robot [v, w]
    a -- noise parameters of the motion model [a1, a2, ..., a6]
    dt -- time interval of prediction
    """

    sigma = np.zeros(5)
    sigma[0] = a[0] * u[0]**2 + a[1] * u[1]**2  # Noise for x
    sigma[1] = a[2] * u[0]**2 + a[3] * u[1]**2  # Noise for y
    sigma[2] = a[4] * u[0]**2 + a[5] * u[1]**2  # Noise for theta
    sigma[3] = a[0] * u[0]**2                   # Noise for linear velocity v
    sigma[4] = a[1] * u[1]**2                   # Noise for angular velocity w

    # Sample noisy velocity commands
    v_hat = u[0] + np.random.normal(0, sigma[0])  # Noisy v
    w_hat = u[1] + np.random.normal(0, sigma[1])  # Noisy w
    v_prime_hat = v_hat + np.random.normal(0, sigma[3])  # Noisy v_prime
    w_prime_hat = w_hat + np.random.normal(0, sigma[4])  # Noisy w_prime

    # Compute the new pose of the robot
    r = v_hat / w_hat if w_hat != 0 else 0  # Avoid division by zero

    # Pose update (x, y, theta)
    x_prime = mu[0] - r * np.sin(mu[2]) + r * np.sin(mu[2] + w_hat * dt)
    y_prime = mu[1] + r * np.cos(mu[2]) - r * np.cos(mu[2] + w_hat * dt)
    theta_prime = mu[2] + w_hat * dt

    # Update velocities (v, w)
    v_prime = v_prime_hat
    w_prime = w_prime_hat

    # Return the new state [x', y', theta', v', w']
    return np.array([x_prime, y_prime, theta_prime, v_prime, w_prime])



from sympy import symbols, Matrix, sin, cos
import numpy as np

def velocity_mm_simpy_five_state():
    """
    Define Jacobian Gt w.r.t state x=[x, y, theta, v, w] and Vt w.r.t command u=[v, w].
    """
    # Symbols for state, controls, and time step
    x, y, theta, v, w, dt = symbols('x y theta v w dt')
    v_cmd, w_cmd = symbols('v_cmd w_cmd')  # Commands

    # Radius of curvature
    R = v_cmd / w_cmd
    beta = theta + w_cmd * dt

    # State update equations (g(u, x))
    gux = Matrix(
        [
            [x - R * sin(theta) + R * sin(beta)],  # x'
            [y + R * cos(theta) - R * cos(beta)],  # y'
            [beta],                                # theta'
            [v_cmd],                               # v'
            [w_cmd],                               # w'
        ]
    )

    # Jacobian Gt (with respect to the state x=[x, y, theta, v, w])
    Gt = gux.jacobian(Matrix([x, y, theta, v, w]))

    # Jacobian Vt (with respect to the commands u=[v_cmd, w_cmd])
    Vt = gux.jacobian(Matrix([v_cmd, w_cmd]))

    # Convert Sympy expressions to NumPy functions for evaluation
    eval_gux = sympy.lambdify((x, y, theta, v_cmd, w_cmd, dt), gux, 'numpy')
    eval_Gt = sympy.lambdify((x, y, theta, v_cmd, w_cmd, dt), Gt, 'numpy')
    eval_Vt = sympy.lambdify((x, y, theta, v_cmd, w_cmd, dt), Vt, 'numpy')

    return eval_gux, eval_Gt, eval_Vt


def landmark_sm_simpy_five_state():
    """
    Defines the landmark measurement model and its Jacobian Ht for a 5D robot state.
    The measurement model outputs [r, phi], where:
    - r is the distance to the landmark
    - phi is the bearing to the landmark relative to the robot's orientation.
    """
    # Symbols for robot pose, velocities, and landmark position
    x, y, theta, v, w, mx, my = symbols("x y theta v w m_x m_y")

    # Measurement model h(x, m)
    r = sympy.sqrt((mx - x)**2 + (my - y)**2)  # Distance
    phi = sympy.atan2(my - y, mx - x) - theta  # Bearing
    hx = Matrix([[r], [phi]])  # Measurement vector [r, phi]

    # Jacobian Ht = dh/dx, with respect to the robot state [x, y, theta, v, w]
    # Velocities v and w do not directly affect the measurement
    state = Matrix([x, y, theta, v, w])
    Ht = hx.jacobian(state)  # This results in a 2x5 Jacobian with 0s for v and w

    # Convert Sympy expressions into NumPy-evaluable functions
    eval_hx = squeeze_sympy_out(sympy.lambdify((x, y, theta, mx, my), hx, "numpy"))
    eval_Ht = squeeze_sympy_out(sympy.lambdify((x, y, theta, mx, my), Ht, "numpy"))

    return eval_hx, eval_Ht
