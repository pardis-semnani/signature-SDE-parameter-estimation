import os
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ControlTerm, Heun, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree
import time
import iisignature
from typing import Callable
from plots import *
import pandas as pd
import json
import itertools
import argparse


def generate_signature_multi_indices(d, q):
    """
    Generate the multi-indices for the signature terms up to level q.

    Parameters:
    d (int): Number of channels (dimensions) in the path.
    q (int): Level of the signature.

    Returns:
    list: A list of tuples representing the multi-indices.
    """
    indices = []
    for level in range(1, q + 1):
        # Generate all possible combinations of indices at this level
        level_indices = itertools.product(range(d), repeat=level)
        indices.extend(level_indices)
    return indices

def generate_all_variable_indices(m: int, q: int) -> dict:
    """
    Generates a mapping from variable names to their indices in the tensors.

    Returns:
    dict: A dictionary where keys are variable names and values are indices arranged.
    from 0 to M - 1 = m+1 + ... + (m+1)^q
    """
    variable_indices = {'Y_0': 0, 't': 1}  # Zero-order term and time variable
    idx = 2  # Starting index for primary variables

    # Level 1 variables (primary variables)
    for i in range(1, m + 1):  # Primary variables indexed from 1 to m
        variable_indices[f'Y_{i}'] = idx
        idx += 1

    # Higher-level variables
    for level in range(2, q + 1):
        level_size = (m + 1) ** level
        for i in range(level_size):
            indices = []
            temp_i = i
            for _ in range(level):
                indices.append(temp_i % (m + 1))
                temp_i = temp_i // (m + 1)
            indices = indices[::-1]  # Reverse to get correct order
            var_name = 'Y_' + '_'.join(str(idx_j) for idx_j in indices)
            variable_indices[var_name] = idx
            idx += 1

    return variable_indices

def generate_variable_indices(m: int, q: int) -> dict:
    """
    Generates a mapping from variable names to their indices in the tensors.

    Returns:
    dict: A dictionary where keys are variable names and values are indices arranged.
    """
    variable_indices = {}  # Initialize empty dict
    idx = 0  # Starting index

    variable_names = ['t'] + [f'Y_{i}' for i in range(1, m + 1)]  # ['t', 'Y_1', 'Y_2']

    # Level 1 variables (primary variables including time)
    for var_name in variable_names:
        variable_indices[var_name] = idx
        idx += 1

    # Higher-level variables
    for level in range(2, q + 1):
        for idx_tuple in itertools.product(range(len(variable_names)), repeat=level):
            var_name = '_'.join(variable_names[i] for i in idx_tuple)
            variable_indices[var_name] = idx
            idx += 1

    return variable_indices


def set_coefficients_from_dict(a, b, variable_indices, coefficients, n):
    """
    Sets the coefficients in the drift and diffusion tensors based on the provided dictionary.

    Parameters:
    a (jnp.ndarray): Drift coefficient matrix.
    b (jnp.ndarray): Diffusion coefficient tensor.
    variable_indices (dict): Mapping of variable names to indices.
    coefficients (dict): Nested dictionary containing the coefficients.
    n (int): Dimension of Brownian motion.
    """
    # Reset coefficients to zero
    a = a.at[:].set(0.0)
    b = b.at[:].set(0.0)

    # Set drift coefficients from drift sub-dictionary
    drift_coeffs = coefficients.get('drift', {})
    for var_name, dependencies in drift_coeffs.items():
        i = variable_indices.get(var_name)
        if i is None:
            continue
        for dep_var, value in dependencies.items():
            k = variable_indices.get(dep_var)
            if k is None:
                continue
            a = a.at[i, k].set(value)

    # Set diffusion coefficients
    diffusion_coeffs = coefficients.get('diffusion', {})
    for var_name, dependencies in diffusion_coeffs.items():
        i = variable_indices.get(var_name)
        if i is None:
            continue
        for dep_var, bm_components in dependencies.items():
            k = variable_indices.get(dep_var)
            if k is None:
                continue
            # Convert keys to int before checking
            for j, value in bm_components.items():
                j_int = int(j)  # Ensure we have an integer index
                if j_int < n:
                    b = b.at[i, j_int, k].set(value)

    return a, b




def compute_total_variables(m: int, q: int) -> (int, list):
    """
    Computes the total number of variables M in the expanded system and the number of variables at each level.
    """
    M = 1  # Zero-order term
    level_sizes = []
    for k in range(1, q + 1):
        size = (m + 1) ** k
        level_sizes.append(size)
        M += size
    return M, level_sizes

def drift_function_with_time(m: int, q: int, a: jnp.ndarray) -> Callable:
    M, level_sizes = compute_total_variables(m, q)
    level_start_indices = [1]  # Start index for first order terms
    for size in level_sizes[:-1]:
        level_start_indices.append(level_start_indices[-1] + size)  # start index for higher order terms

    def drift(t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        drift_vector = jnp.zeros(M)
        drift_vector = drift_vector.at[0].set(0.0)  # Zero-order term remains constant
        drift_vector = drift_vector.at[1].set(1.0)  # Time variable derivative is 1

        # Level 1 (primary variables)
        for i in range(2, m + 2):  # Indices for true primary variables are 2 to m+1 (0-indexed)
            sum_a_y = jnp.dot(a[i], y) # compute linear drift
            drift_vector = drift_vector.at[i].set(sum_a_y)

        # Drift for higher levels are defined recursively
        for level in range(2, q + 1):
            curr_level_start = level_start_indices[level - 1]
            curr_level_size = level_sizes[level - 1] # -1 because zeroth order is omitted

            for idx in range(curr_level_size):
                # Multi-index for current variable
                indices = []
                temp_idx = idx
                for _ in range(level):
                    indices.append(temp_idx % (m + 1))
                    temp_idx = temp_idx // (m + 1)
                indices = indices[::-1]  # Reverse to get correct order
                i = curr_level_start + idx  # Index of current variable

                # Compute product term
                product_term = drift_vector[1 + indices[-1]]
                for k in indices[:-1]:
                    product_term *= y[1 + k]
                sum_a_y = jnp.dot(a[i], y)
                drift_vector = drift_vector.at[i].set(product_term + sum_a_y)
        return drift_vector

    return drift

def diffusion_function_with_time(m: int, n: int, q: int, b: jnp.ndarray) -> Callable:
    M, level_sizes = compute_total_variables(m, q)
    level_start_indices = [1]
    for size in level_sizes[:-1]:
        level_start_indices.append(level_start_indices[-1] + size)

    def diffusion(t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        diffusion_matrix = jnp.zeros((M, n))
        diffusion_matrix = diffusion_matrix.at[0, :].set(0.0)  # Zero-order term
        diffusion_matrix = diffusion_matrix.at[1, :].set(0.0)  # Time variable

        # Level 1 (primary variables)
        for i in range(2, m + 2):
            for j in range(n):
                sum_b_y = jnp.dot(b[i, j], y)
                diffusion_matrix = diffusion_matrix.at[i, j].set(sum_b_y)

        # Higher levels
        for level in range(2, q + 1):
            curr_level_start = level_start_indices[level - 1]
            curr_level_size = level_sizes[level - 1]

            for idx in range(curr_level_size):
                # Multi-index for current variable
                indices = []
                temp_idx = idx
                for _ in range(level):
                    indices.append(temp_idx % (m + 1))
                    temp_idx = temp_idx // (m + 1)
                indices = indices[::-1]  # Reverse to get correct order
                i = curr_level_start + idx

                for j in range(n):
                    # Compute product term
                    product_term = diffusion_matrix[1 + indices[-1], j]
                    for k in indices[:-1]:
                        product_term *= y[1 + k]
                    sum_b_y = jnp.dot(b[i, j], y)
                    diffusion_value = product_term + sum_b_y
                    diffusion_matrix = diffusion_matrix.at[i, j].set(diffusion_value)
        return diffusion_matrix

    return diffusion

def compute_iterated_integrals_iisignature(primary_variables: np.ndarray, q: int) -> np.ndarray:
    """
    Computes iterated integrals up to level q using iisignature.

    Parameters:
    primary_variables (np.ndarray): Array of shape (T, d) containing primary variable trajectories.
    q (int): Level of iterated integrals to compute.

    Returns:
    np.ndarray: Computed iterated integrals (signature) up to level q.
    """
    # iisignature expects the path to be an array of shape (T, d)
    # Ensure primary_variables includes the time variable as the first column
    path = primary_variables
    sig = iisignature.sig(path, q)
    return sig

def solve_sde(
    drift_function: Callable,
    diffusion_function: Callable,
    initial_condition: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    key: jax.random.PRNGKey  # Added key parameter
):
    # Define Brownian motion
    n = diffusion_function(0, initial_condition, None).shape[1]  # Dimension of Brownian motion
    bm = VirtualBrownianTree(
        t0=t0,
        t1=t1,
        tol=1e-3,
        shape=(n,),
        key=key
    )


    # Define the SDE terms
    terms = MultiTerm(
        ODETerm(drift_function),
        ControlTerm(diffusion_function, bm)
    )

    # Set the solver
    solver = Heun()  # Heun solver, suitable for Stratonovich SDEs with non-commutative noise

    # Solve the SDE
    num_steps = int((t1 - t0) / dt) + 1  # Ensure the endpoint t1 is included
    ts = jnp.linspace(t0, t1, num_steps)
    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(
        terms,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=initial_condition,
        saveat=saveat
    )

    return sol, solver

def estimate_A(X, dt, pinv=False):
    """
    Calculate the approximate closed form estimator A_hat for time homogeneous linear drift from multiple trajectories

    Parameters:
        trajectories (numpy.ndarray): 3D array (num_trajectories, num_steps, d),
        where each slice corresponds to a single trajectory.
        dt (float): Discretization time step.
        pinv: whether to use pseudo-inverse. Otherwise, we use left_Var_Equation

    Returns:
        numpy.ndarray: Estimated drift matrix A given the set of trajectories
    """
    num_trajectories, num_steps, d = X.shape
    sum_Edxt_Ext = np.zeros((d, d))
    sum_Ext_ExtT = np.zeros((d, d))
    for t in range(num_steps - 1):
        sum_dxt_xt = np.zeros((d, d))
        sum_xt_xt = np.zeros((d, d))
        for n in range(num_trajectories):
            xt = X[n, t, :]
            dxt = X[n, t + 1, :] - X[n, t, :]
            sum_dxt_xt += np.outer(dxt, xt)
            sum_xt_xt += np.outer(xt, xt)
        sum_Edxt_Ext += sum_dxt_xt / num_trajectories
        sum_Ext_ExtT += sum_xt_xt / num_trajectories

    return np.matmul(sum_Edxt_Ext, np.linalg.pinv(sum_Ext_ExtT)) * (1 / dt)


def estimate_GGT(trajectories, T, est_A=None):
    """
    Estimate the observational diffusion GG^T for a multidimensional linear
    additive noise SDE from multiple trajectories

    Parameters:
        trajectories (numpy.ndarray): 3D array (num_trajectories, num_steps, d),
        where each "slice" (2D array) corresponds to a single trajectory.
        T (float): Total time period.
        est_A (numpy.ndarray, optional): pre-estimated drift A.
        If none provided, est_A = 0, modeling a pure diffusion process

    Returns:
        numpy.ndarray: Estimated GG^T matrix.
    """
    num_trajectories, num_steps, d = trajectories.shape
    dt = T / (num_steps - 1)

    # Initialize the GG^T matrix
    GGT = np.zeros((d, d))

    if est_A is None:
        # Compute increments ΔX for each trajectory (no drift adjustment)
        increments = np.diff(trajectories, axis=1)
    else:
        # Adjust increments by subtracting the deterministic drift: ΔX - A * X_t * dt
        increments = np.diff(trajectories, axis=1) - dt * np.einsum('ij,nkj->nki', est_A, trajectories[:, :-1, :])

    # Sum up the products of increments for each dimension pair across all trajectories and steps
    for i in range(d):
        for j in range(d):
            GGT[i, j] = np.sum(increments[:, :, i] * increments[:, :, j])

    # Divide by total time T*num_trajectories to normalize
    GGT /= T * num_trajectories
    return GGT



def run_experiment(config: dict, experiment_name: str):
    t_0 = time.time()

    N = config["N"]
    m = config["m"]
    n = config["n"]
    q = config["q"]
    q_iterated = config["q_iterated"]
    t1 = config["t1"]
    dt = config["dt"]
    coefficients = config["coefficients"]
    base_key_val = config["base_key"]
    n_seeds = config.get("n_seeds", 1)
    plotting_on = config.get("plotting_on", 0)
    # Initialize a single DataFrame to collect results from all seeds
    all_results = []
    output_dir = experiment_name

    for seed in range(n_seeds):
        t_1 = time.time()
        current_key = base_key_val + seed
        t0 = 0.0

        M, level_sizes = compute_total_variables(m, q)
        X0 = jnp.zeros(M)
        X0 = X0.at[0].set(1.0)
        X0 = X0.at[1].set(0.0)

        a = jnp.zeros((M, M))
        b = jnp.zeros((M, n, M))

        variable_indices_full = generate_all_variable_indices(m, q)
        a, b = set_coefficients_from_dict(a, b, variable_indices_full, coefficients, n)

        drift = drift_function_with_time(m, q, a)
        diffusion = diffusion_function_with_time(m, n, q, b)

        base_key = jax.random.PRNGKey(current_key)
        keys = jax.random.split(base_key, N)

        solutions = []
        for i in range(N):
            sol, solver_used = solve_sde(
                drift_function=drift,
                diffusion_function=diffusion,
                initial_condition=X0,
                t0=t0,
                t1=t1,
                dt=dt,
                key=keys[i]
            )
            solutions.append(sol)

        ys_array = jnp.stack([sol.ys for sol in solutions], axis=0)
        ys_array = np.array(ys_array)


        primary_data_with_time = ys_array[:, :, 1:m + 2]

        sig_length = iisignature.siglength(m + 1, q_iterated)
        signatures = np.zeros((N, sig_length))
        for n_ in range(N):
            path = primary_data_with_time[n_, :, :]
            sig = iisignature.sig(path, q_iterated)
            signatures[n_, :] = sig

        expected_signature = np.mean(signatures, axis=0)
        expected_signature_with_empty = np.concatenate(([1.0], expected_signature))

        variable_names = ['t'] + [f'Y_{i}' for i in range(1, m + 1)]
        multi_indices = generate_signature_multi_indices(m+1, q_iterated)
        variable_indices = generate_variable_indices(m, q_iterated)

        multi_index_strings = ['{}']
        multi_index_to_variable_index = {}
        for idx_tuple in multi_indices:
            variable_name = '_'.join(variable_names[idx_j] for idx_j in idx_tuple)
            variable_index = variable_indices.get(variable_name)
            multi_index_strings.append(variable_name)
            multi_index_to_variable_index[variable_name] = variable_index

        expected_value_sde = []
        for multi_idx_str in multi_index_strings:
            variable_index = multi_index_to_variable_index.get(multi_idx_str)
            if variable_index is not None and variable_index < ys_array.shape[2]:
                variable_values = ys_array[:, -1, 1 + variable_index]
                expected_value = np.mean(variable_values)
            elif multi_idx_str == '{}':
                expected_value = 1.0
            else:
                expected_value = np.nan
            expected_value_sde.append(expected_value)


        if len(expected_value_sde) != len(multi_index_strings):
            print(len(expected_value_sde))
            print(len(multi_index_strings))
            raise ValueError("Length mismatch: `expected_value_sde` and `multi_index_strings` must be the same length.")

        df = pd.DataFrame({
            'multi_index': multi_index_strings,
            'expected_value': expected_signature_with_empty,
            'expected_value_sde': expected_value_sde
        })
        df['level'] = [0] + [len(idx_tuple) for idx_tuple in multi_indices]
        df['difference'] = df['expected_value'] - df['expected_value_sde']

        all_results.append(df)
        t_2 = time.time()
        print(
            f'Computation time for seed {seed + 1} of {n_seeds} : {t_2 - t_1}. Cumulative computation time: {t_2 - t_0}')
        #
        if plotting_on != 0:
            plot_marginals(solutions, m, experiment_name = experiment_name, output_dir = output_dir)
            solution = solutions[0]
            plot_primary_variables(solution, m, experiment_name = experiment_name, output_dir = output_dir)
            plot_all_variables(solution, experiment_name = experiment_name, output_dir = output_dir)
            plt.close()
            plot_differences(df)

    final_df = pd.concat(all_results, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)

    # Save the CSV file
    csv_filename = os.path.join(output_dir, f"{experiment_name}_expected_signatures_N-{N}_seed-{base_key_val}_n_seeds-{n_seeds}.csv")
    final_df.to_csv(csv_filename, index=False)

    # Prepare and save the JSON metadata
    metadata = {
        'N': N,
        'm': m,
        'n': n,
        'q': q,
        'q_iterated': q_iterated,
        't1': t1,
        'dt': dt,
        'coefficients': coefficients,
        'base_key': int(base_key[0]),
        'n_seeds': n_seeds,
        'solver': solver_used.__class__.__name__
    }

    json_filename = os.path.join(output_dir, f"{experiment_name}_metadata_N-{N}_seed-{base_key_val}.json")
    with open(json_filename, 'w') as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument("--experiment_name", required=True, help="Base name for experiment.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    run_experiment(config, args.experiment_name)
