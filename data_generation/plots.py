import numpy as np
import matplotlib.pyplot as plt
import os


def plot_marginals(solutions, m, experiment_name, output_dir):
    # Collect all time points and corresponding values
    t_values = np.array(solutions[0].ts)  # Assume all time series have the same time points
    y_values_all = []  # To store all y-values for each time series

    for solution in solutions:
        y_values_all.append(np.array(solution.ys[:, 2:m + 2]))

    y_values_all = np.array(y_values_all)  # Shape: (N, T, d)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.arange(m))  # Generate distinct colors for each dimension

    for dim in range(m):
        for n in range(len(solutions)):  # Overlay dots for each time series
            plt.scatter(t_values, y_values_all[n, :, dim], color=colors[dim], s=5, alpha=0.6)
        plt.plot([], [], color=colors[dim], label=f'Y_{dim + 1}')  # Add to legend

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Marginal Distributions for {experiment_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_marginals_plot.png"))
    plt.show()


# Function to plot strictly primary variables
def plot_primary_variables(solution, m, experiment_name, output_dir):
    t_values = np.array(solution.ts)
    y_values = np.array(solution.ys[:, 2:m+2])  # Extract primary variables (indices 2 to m+1)

    plt.figure(figsize=(10, 6))
    for i in range(y_values.shape[1]):
        plt.plot(t_values, y_values[:, i], label=f'Y_{i+1}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Primary Variables Trajectories for {experiment_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_primary_variables_plot.png"))
    plt.show()

def plot_differences(df):
    levels = sorted(df['level'].unique())
    for level in levels:
        df_level = df[df['level'] == level]
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(df_level)), df_level['difference'].abs(), tick_label=df_level['multi_index'])
        plt.title(f'Absolute Differences at Level {level}')
        plt.xlabel('Multi-Index')
        plt.ylabel('Absolute Difference')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

# Function to plot all variables
def plot_all_variables(solution, experiment_name, output_dir):
    t_values = np.array(solution.ts)
    y_values = np.array(solution.ys)  # All variables

    plt.figure(figsize=(10, 6))
    for i in range(y_values.shape[1]):
        plt.plot(t_values, y_values[:, i], label=f'Y_{i}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('All Variables Trajectories')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_all_variables_plot.png"))
    plt.show()

def plot_iterated_integrals(iterated_integrals_higher, higher_order_variables, min_length, time_steps):
    plt.figure(figsize=(12, 8))
    for idx in range(min_length):
        plt.plot(time_steps, higher_order_variables[:, idx], linestyle='-', label=f'Expanded Var {idx}')
        plt.plot(time_steps, iterated_integrals_higher[:, idx], linestyle='--', label=f'Numerical Var {idx}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Comparison of Expanded System and Numerical Integration')
    plt.legend(ncol=2, loc='upper left', fontsize='small')
    plt.grid(True)
    plt.show()