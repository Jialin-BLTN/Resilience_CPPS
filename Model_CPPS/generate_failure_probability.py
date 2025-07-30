import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp
import pandas as pd
from create_net import create_network
from sklearn.preprocessing import MinMaxScaler
from line_flow_in_contingency import get_line_flow_after_contingency
import math

"""
This script is designed to analyze line failures in a power grid by calculating conditional failure probabilities, 
evaluating line loading changes, and visualizing the results.
The output will be the conditional failure probabilities, line loading changes, and a plot of these values.
"""


def calculate_failure_probabilities(n, rq, initial_p_interactions, e=1e-6):
    """
    Calculate the failure probabilities of system components based on given initial and subsequent interactions. Rui's paper.

    Parameters
    ----------
    n : int
        Total number of system components.
    rq : list of int
        List representing the operating conditions of each system component.
    initial_p_interactions : list of np.array
        List of numpy arrays representing the initial and subsequent interactions.
    e : float, optional
        Threshold for comparing failure probabilities, by default 1e-6.

    Returns
    -------
    np.array
        Final failure probabilities of each system component.
    """
    # Step 1: Initialize V1
    V1 = np.zeros(n)
    for i in range(n):
        if rq[i] == 1:
            V1[i] = 1

    V1_prime = np.ones(n) - V1

    G = 2
    while True:
        V2 = np.ones(n)
        for i in range(n):
            prod_term = 1
            for j in range(n):
                if V1[j] != 0:
                    prod_term *= (1 - V1[j] * initial_p_interactions[0][i, j])
            V2[i] = 1 - prod_term

        V2_prime = V1_prime * (1 - V2)

        if np.any(V2 > e):
            V1 = V2
            V1_prime = V2_prime
            G += 1
        else:
            break

    V_final = np.ones(n)
    for i in range(n):
        prod_term = 1
        for k in range(G):
            prod_term *= (1 - V1[i])
        V_final[i] = 1 - prod_term

    return V_final

def top_n_values_indices(arr, n):
    """
    Get the top n values and their indices from a 2D array.

    Parameters:
    arr (numpy array): Input 2D array.
    n (int): Number of top values to retrieve.

    Returns:
    tuple: Top n values and their 2D indices.
    """
    flat_arr = arr.flatten()
    indices_of_top_n = np.argpartition(flat_arr, -n)[-n:]
    indices_of_top_n_sorted = indices_of_top_n[np.argsort(-flat_arr[indices_of_top_n])]
    top_n_indices_2d = np.unravel_index(indices_of_top_n_sorted, arr.shape)
    top_n_values = flat_arr[indices_of_top_n_sorted]

    return top_n_values, top_n_indices_2d

def map_values(x, M, dampening_factor):
    """
    Apply a mapping function with dampening for values above a threshold.

    Parameters:
    x (list or numpy array): Input values.
    M (float): Maximum value for the sigmoid function.
    dampening_factor (float): Factor to dampen the increase above the threshold.

    Returns:
    numpy array: Mapped values.
    """
    x = np.array(x)
    threshold = 5000
    threshold_value = 1 / (1 + np.exp(-10 / M * (threshold - M / 2)))

    return np.where(x <= threshold,
                    1 / (1 + np.exp(-10 / M * (x - M / 2))),
                    threshold_value + dampening_factor * (1 / (1 + np.exp(-10 / M * (x - M / 2))) - threshold_value))



def compute_arrays(X, disconnect_line_0):
    """
    Compute the inverse distance arrays for a given disconnect line.

    Parameters:
    X (numpy array): Input data.
    disconnect_line_0 (int): Index of the disconnect line.

    Returns:
    list: Computed inverse distance arrays.
    """
    epsilon = 1e-8
    arr_0 = [1.0 / (abs(X[i] - X[disconnect_line_0]) + epsilon) for i in range(len(X))]
    return np.concatenate(arr_0).tolist()

def run_model(hour, failure_time, failure_list, visualization,cnt):
    # Load data


    # Generate base case data (normal condition, before failure happens)
    gen = pd.read_csv("..\\gen_118.csv", delimiter=',')
    load = pd.read_csv("..\\load_118.csv", delimiter=',')

    gen = gen.set_index('SCED Time Stamp')
    load = load.set_index('Hour Ending')

    # load = load/2
    gen['5'] = gen['5'] * 3
    gen['25'] = gen['25'] * 3
    gen['64'] = gen['64'] * 2
    gen['69'] = gen['69'] * 2
    gen['3'] = 200
    gen['53'] = 300
    gen['76'] = 300

    pv_list = ['5', '25', '39', '64', '69', '88', '98', '102']
    # gen[pv_list] = gen[pv_list] +20

    gen.columns = [f'gen{i}' for i in range(53)]
    load.columns = [f'load{i}' for i in range(99)]

    index = f"2023-08-11 {hour:02}:00:00"
    load_data = load[load.index == index]
    gen_data = gen[gen.index == index]

    net = create_network(load_data, gen_data)

    pp.runopp(net, delta=1e-16)

    net.gen['p_mw'] = net.res_gen['p_mw']
    net.gen['vm_pu'] = net.res_gen['vm_pu']
    base_case = net.res_line['loading_percent'].copy()

    #base_case = pd.read_csv(f"Loading_{hour}.csv")
    before_case = base_case.values.flatten()







    # Get line flow after contingency
    after_line = get_line_flow_after_contingency(failure_time, failure_list,cnt)



    if visualization == True:
        # Plot the results
        plt.figure(figsize=(36, 10))

        plt.bar(range(len(after_line)), after_line, label="Loading ratios after contingency happens")
        plt.xticks(np.arange(0, 193, 4), fontsize=16)  # Setting x-ticks to show every 4 steps
        plt.xlim(-1, len(after_line))  # Set the limits of the x-axis with a slight buffer

        plt.legend(fontsize=16)
        plt.grid(axis='y')  # Grid lines for the y-axis
        plt.xlabel("Line number ", fontsize=20)
        plt.ylabel("(%) ", fontsize=20)
        plt.tight_layout()  # Adjust layout to not overlap
        plt.show()

    return  after_line