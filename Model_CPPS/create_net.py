import pandapower as pp
import pandas as pd
import numpy as np
from pandapower.timeseries import DFData, OutputWriter
from pandapower.control import ConstControl
import matplotlib.pyplot as plt
import time
import itertools
import warnings

warnings.filterwarnings("ignore")

"""
This script is designed to create a power network using the pandapower library, configure it with various parameters,
and return the configured network. The network setup includes creating new lines, adjusting generation and load data,
and modifying line parameters to simulate different scenarios.

The output will be the configured pandapower network object.
"""


def create_network(load_data, gen_data):
    """
    Create and configure a power network based on the IEEE 118-bus system.

    Parameters:
    load_data (DataFrame): Load data for the network.
    gen_data (DataFrame): Generation data for the network.

    Returns:
    net (pandapowerNet): Configured pandapower network.
    """
    # Initialize the IEEE 118-bus system
    net = pp.networks.case118()

    # Lists of buses with PV and wind generation
    pv_list = [5, 25, 33, 64, 69, 88, 98, 102]
    wind_list = [14, 17, 54, 55, 84, 86]
    pv_element = [index for index, bus in enumerate(net.gen['bus']) if bus in pv_list]
    wind_element = [index for index, bus in enumerate(net.gen['bus']) if bus in wind_list]

    # Adjust cost parameters for PV and wind generation
    net.poly_cost.loc[net.poly_cost['element'].isin(pv_element), 'cp1_eur_per_mw'] = 1
    net.poly_cost.loc[net.poly_cost['element'].isin(pv_element), 'cp2_eur_per_mw2'] = 0
    net.poly_cost.loc[net.poly_cost['element'].isin(wind_element), 'cp1_eur_per_mw'] = 1
    net.poly_cost.loc[net.poly_cost['element'].isin(wind_element), 'cp2_eur_per_mw2'] = 0

    # Set line and bus parameters
    net.line['max_i_ka'] = 1
    net.line['max_loading_percent'] = 100
    net.bus['max_vm_pu'] = 1.05
    net.bus['min_vm_pu'] = 0.95

    # Set load and generation data
    net.load['p_mw'] = load_data.values[0]
    net.gen['max_p_mw'] = gen_data.values[0]

    return net
