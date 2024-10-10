import jax
import jax.numpy as jnp
from jax import jit, jacfwd

@jit
def dynamics(state, inputs, g, m):
    x, y, z, vx, vy, vz, roll, pitch, yaw = state
    curr_thrust, curr_rolldot, curr_pitchdot, curr_yawdot = inputs

    sr = jnp.sin(roll)
    sy = jnp.sin(yaw)
    sp = jnp.sin(pitch)
    cr = jnp.cos(roll)
    cp = jnp.cos(pitch)
    cy = jnp.cos(yaw)

    vxdot = -(curr_thrust / m) * (sr * sy + cr * cy * sp)
    vydot = -(curr_thrust / m) * (cr * sy * sp - cy * sr)
    vzdot = g - (curr_thrust / m) * (cr * cp)

    return jnp.array([vx, vy, vz, vxdot, vydot, vzdot, curr_rolldot, curr_pitchdot, curr_yawdot])

# Function to integrate dynamics over time
@jit
def integrate_dynamics(state, inputs, integration_step, integrations_int, g, m):
    def for_function(i, current_state):
        return current_state + dynamics(current_state, inputs, g, m) * integration_step

    pred_state = jax.lax.fori_loop(0, integrations_int, for_function, state)
    return pred_state

# Prediction function
@jit
def predict_states(state, last_input, T_lookahead, g, m, integration_step=0.1):
    inputs = last_input.flatten()
    integrations_int = 8 #int(T_lookahead / integration_step)
    pred_state = integrate_dynamics(state, inputs, integration_step, integrations_int, g, m)
    return pred_state

# Prediction function
@jit
def predict_outputs(state, last_input, T_lookahead, g, m, C, integration_step=0.1):
    inputs = last_input.flatten()
    integrations_int = 8 #int(T_lookahead / integration_step)
    pred_state = integrate_dynamics(state, inputs, integration_step, integrations_int, g, m)
    return C@pred_state

# Compute Jacobian
@jit
def compute_jacobian(state, last_input, T_lookahead, g, m, C, integration_step):
    jac_fn = jacfwd(lambda x: predict_outputs(state, x, T_lookahead, g, m, C, integration_step))
    return jac_fn(last_input)



# Compute adjusted inverse Jacobian
@jit
def compute_adjusted_invjac(state, last_input, T_lookahead, g, m, C, integration_step):
    jac = compute_jacobian(state, last_input, T_lookahead, g, m, C, integration_step)
    inv_jacobian = jnp.linalg.pinv(jac)
    inv_jacobian_modified = inv_jacobian.at[:, 2].set(-inv_jacobian[:, 2])
    return inv_jacobian_modified
    # return jac
