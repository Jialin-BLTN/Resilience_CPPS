import numpy as np
import pandas as pd
import copy
import random
import openpyxl

from generate_failure_probability import run_model
import os
### this is the time when failure happens
failure_time = "2023-08-11 19:00:00"

# this is the hour when failure happens
hour = 19

### this is the failure list, in the example, initial failures are lines #68,69, and line #175 fails in the first generation.
# failure_list = [[68,69],[175]]
# decide whether to plot the results or not. True or False
visualization = False

cnt=0
# 2 here means how many cases
for m in range(2):
    failure_list = [[random.randint(0, 192), random.randint(0, 192), random.randint(0, 192), random.randint(0, 192)]]

    initial_fail = copy.deepcopy(failure_list)
    print("initial_failure: ", initial_fail)
    contin = 1
    cnt = cnt + 1
    # print(cnt)

    while contin == 1:
        line_loading_after_fails = run_model(hour, failure_time, initial_fail,
                                                                              visualization, cnt)

        for i in range(len(line_loading_after_fails)):
            if line_loading_after_fails[i] > 99:
                failure_list.append([i])

        if len(initial_fail) == len(failure_list):
            contin = 0
        else:
            initial_fail = failure_list
    print(failure_list)
    print("Failure propagation path: ", failure_list)


    all_line = [item for sublist in failure_list for item in sublist]

    df1 = pd.DataFrame(all_line, )

    directory = r'output'
    filename = f'failure_results{cnt}.xlsx'
    full_path = os.path.join(directory, filename)

    # df1.to_excel("failure_results.xlsx", sheet_name='Sheet 3')
    # with pd.ExcelWriter(f'failure_results{cnt}.xlsx', engine='openpyxl', mode='a') as writer:
    with pd.ExcelWriter(full_path, engine='openpyxl', mode='a') as writer:

        df1.to_excel(writer, sheet_name='lines failed', index=False, header=None)

# print(f"failed lines= {all_line}")

# now put all the results into one file so they can eventually be combined for multiple failures.
