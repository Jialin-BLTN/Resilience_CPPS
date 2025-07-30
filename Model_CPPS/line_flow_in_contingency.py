import pandapower as pp
import pandas as pd
import numpy as np
import os
import openpyxl
from create_net import create_network
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.control import ConstControl
import matplotlib.pyplot as plt
import time
import itertools
import warnings
warnings.filterwarnings("ignore")
try:
    import seaborn
    colors = seaborn.color_palette()
except:
    colors = ["b", "g", "r", "c", "y"]

def create_data_source(load,gen):
    profiles = pd.concat([load.reset_index(drop=True),gen.reset_index(drop=True)], axis=1)
    ds = DFData(profiles)
    return profiles, ds

def create_controllers(net, ds, load,gen):
    ConstControl(net, element='load', variable='p_mw', element_index=net.load.index,
                 data_source=ds, profile_name=load.columns.tolist())

    ConstControl(net, element='gen', variable='max_p_mw', element_index=net.gen.index,
                 data_source=ds, profile_name=gen.columns.tolist())

def create_output_writer(net, time_steps, output_dir):
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xlsx", log_variables=list())
    ow.log_variable('res_sgen', 'p_mw')
    ow.log_variable('res_gen', 'p_mw')
    #ow.log_variable('res_cost', 'value')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'i_ka')
    return ow
def handle_24_hour(time_str):
    if '24:00' in time_str:
        date, time = time_str.split()
        time_str = pd.Timestamp(date) + pd.Timedelta(days=1)
    else:
        time_str = pd.Timestamp(time_str)
    return time_str


def get_line_flow_after_contingency(index, disconnected, cnt):
    gen = pd.read_csv("..\\gen_118.csv", delimiter=',')
    load = pd.read_csv("..\\load_118.csv", delimiter=',')

    # gen = gen.tail(360)
    # load = load.tail(360)

    gen = gen.set_index('SCED Time Stamp')
    load = load.set_index('Hour Ending')

    #load = load/2
    gen['5'] = gen['5'] *3
    gen['25'] = gen['25'] *3
    gen['64'] = gen['64'] *2
    gen['69'] = gen['69'] *2
    gen['3'] =200
    gen['53'] = 300
    gen['76'] = 300

    pv_list = ['5', '25', '33', '64', '69', '88', '98', '102']
    #gen[pv_list] = gen[pv_list] +20

    gen.columns =  [f'gen{i}' for i in range(53)]
    load.columns =  [f'load{i}' for i in range(99)]


    flag = 0
    results_dataframes = [pd.DataFrame() for _ in range(193)]

    #8/11/2023 0:00

    load_data = load[load.index == index]
    gen_data = gen[gen.index == index]

    net = create_network(load_data, gen_data)

    pp.runopp(net, delta=1e-16)

    net.gen['p_mw'] = net.res_gen['p_mw']
    net.gen['vm_pu'] = net.res_gen['vm_pu']
    initial_load = net.res_load['p_mw'].values.copy()
    # added this to also output the initial loading percents before the contingency happens"""
    initial_loading= net.res_line['loading_percent']
    net.line['in_service'] = True
    #print(disconnected[0])
    net.line.loc[disconnected[0], 'in_service'] = False
    pp.runpp(net, algorithm='nr', init='results', max_iteration=30, numba=False)

    line_status = np.ones(193)
    for i in range(len(net.line)):
        if (net.line.loc[i, 'in_service'] == False):
            line_status[i] = 0

    for i in range(len(disconnected)):
        if i == 0:
            continue
        for j in disconnected[i]:
            #print(i,j)
            #print(dis)
            net.line.loc[j, 'in_service'] = False
            #print(j)
        pp.runpp(net, algorithm='nr', init='results', max_iteration=30, numba=False)
    indices = net.res_load[net.res_load['p_mw'] == 0].index.tolist()

    load_shed_index = []
    load_shed_amount= []
    for i in indices:
        print(f"Load shed for load #{i} with {initial_load[i]} MW")
        load_shed_index.append([i])
        load_shed_amount.append([initial_load[i]])

   # print (load_shed_index)
    #np.savetxt(f'load_shed_bus.csv', load_shed_index, delimiter=',')

    #df1 = pd.DataFrame(load_shed_index, index=['load shed bus'],
    #               columns=['failure 1'])
    #df2 = pd.DataFrame(load_shed_amount, index=['load shed amount'],
     #              columns=['failure 1'])
    df1 = pd.DataFrame(load_shed_index)
    df2 = pd.DataFrame(load_shed_amount)
    df3 = initial_loading
    df4 = pd.DataFrame(line_status)

    directory = r'output'
    filename = f'failure_results{cnt}.xlsx'
    full_path = os.path.join(directory, filename)

    with pd.ExcelWriter(full_path) as writer:
        df1.to_excel(writer, sheet_name='load not served bus')
        df2.to_excel(writer, sheet_name='load not served MW')
        df3.to_excel(writer, sheet_name='Initial line loading')
        df4.to_excel(writer, sheet_name='Initial line status')
   # with pd.ExcelWriter('failure_results.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    #    df1.to_excel(writer, sheet_name='Sheet1', index=False, header=None)
    #    df2.to_excel(writer, sheet_name='Sheet2', index=False, header=None)

    return net.res_line['loading_percent']



    # if case_number == 5:
    #     break
# i = 0
# for line_number in range(193):
#     print(line_number)
#     results_dataframes[i].to_csv(f"N-1_results_current\\N-1_Line_{line_number}.csv", index=False)
#     i += 1
