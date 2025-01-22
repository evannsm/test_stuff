import jax.numpy as jnp
from jax import jit, jacfwd, lax, jacrev, hessian

GRAVITY = 9.806
C = jnp.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1]])

@jit
def dynamics(state, input, mass):
    """Compute the state derivative."""
    x, y, z, vx, vy, vz, roll, pitch, yaw = state
    curr_thrust, curr_rolldot, curr_pitchdot, curr_yawdot = input

    sr = jnp.sin(roll)
    sy = jnp.sin(yaw)
    sp = jnp.sin(pitch)
    cr = jnp.cos(roll)
    cp = jnp.cos(pitch)
    cy = jnp.cos(yaw)

    vxdot = -(curr_thrust / mass) * (sr * sy + cr * cy * sp)
    vydot = -(curr_thrust / mass) * (cr * sy * sp - cy * sr)
    vzdot = GRAVITY - (curr_thrust / mass) * (cr * cp)

    xdot = jnp.array([vx, vy, vz, vxdot, vydot, vzdot, curr_rolldot, curr_pitchdot, curr_yawdot]).reshape((9,1))
    return xdot

@jit
def fwd_euler(state, input, integration_step, integrations_int, mass):
    """Forward Euler integration."""
    def for_function(i, current_state):
        return current_state + dynamics(current_state, input, mass) * integration_step

    pred_state = lax.fori_loop(0, integrations_int, for_function, state)
    return pred_state

@jit
def predict_state(state, u, T_lookahead, integration_step, mass):
    """Predict the next state."""
    integrations_int = (T_lookahead / integration_step).astype(int)
    pred_state = fwd_euler(state, u, integration_step, integrations_int, mass)
    return pred_state

@jit
def predict_output(state, u, T_lookahead, integration_step, mass):
    """Predict the output."""
    pred_state = predict_state(state, u, T_lookahead, integration_step, mass)
    return C @ pred_state

@jit
def get_jac_pred_u(state, last_input, T_lookahead, integration_step, mass):
    raw_val = jacfwd(predict_output, 1)(state, last_input, T_lookahead, integration_step, mass)
    return raw_val.reshape((4,4))

@jit
def get_inv_jac_pred_u(state, last_input, T_lookahead, integration_step, mass):
    return jnp.linalg.pinv(get_jac_pred_u(state, last_input, T_lookahead, integration_step, mass).reshape((4,4)))
@jit
def NR_tracker_original(currstate, currinput, ref, T_lookahead, integration_step, sim_step, mass):
    """Newton-Raphson method to track the reference trajectory."""
    alpha = jnp.array([20, 30, 30, 30]).reshape((4,1))
    pred = predict_output(currstate, currinput, T_lookahead, integration_step, mass)
    error = (ref - pred)
    dgdu = get_jac_pred_u(currstate, currinput, T_lookahead, integration_step, mass)
    dgdu_inv = jnp.linalg.inv(dgdu)
    NR = dgdu_inv @ error
    udot = alpha * NR
    change_u = udot * sim_step
    new_u = currinput + change_u
    return new_u

@jit
def error_func(state, u, T_lookahead, integration_step, ref, mass):
    """Compute the error."""
    pred = predict_output(state, u, T_lookahead, integration_step, mass)
    return jnp.sum((ref - pred)**2)

@jit
def get_grad_error_u(state, u, T_lookahead, integration_step, ref, mass):
    return jacfwd(error_func, 1)(state, u, T_lookahead, integration_step, ref, mass)

@jit
def get_hess_error_u(state, u, T_lookahead, integration_step, ref, mass):
    return jacfwd(get_grad_error_u, 1)(state, u, T_lookahead, integration_step, ref, mass).reshape(4,4)

@jit
def NR_tracker_optim(currstate, currinput, ref, T_lookahead, integration_step, sim_step, mass):
    """Newton-Raphson method to track the reference trajectory."""
    alpha = jnp.array([20, 30, 30, 30]).reshape((4,1))
    # pred = predict_output(currstate, currinput, T_lookahead, integration_step, mass)
    # error = error_func(currstate, currinput, T_lookahead, integration_step, ref, mass)
    dgdu = get_grad_error_u(currstate, currinput, T_lookahead, integration_step, ref, mass)
    hess = get_hess_error_u(currstate, currinput, T_lookahead, integration_step, ref, mass)
    d2gdu2_inv = jnp.linalg.inv(hess)
    scaling = d2gdu2_inv @ dgdu
    NR = scaling
    udot = -alpha * NR
    change_u = udot * sim_step
    new_u = currinput + change_u
    return new_u
