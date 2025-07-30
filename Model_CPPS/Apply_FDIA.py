import pandapower as pp
import pandapower.networks as pn
import pandapower.estimation as est
import pandapower.create as create
import pandapower.topology as top
import numpy as np

#  Load IEEE 118-bus system
net = pn.case118()

# Reinforce the original network to meet N-1 contingency by ensuring there's no isolated bus when one line fails
# This is a simple check for radial lines that would isolate buses
critical_lines = []
for i in range(len(net.line)):
    net_temp = net.deepcopy()
    net_temp.line.drop(i, inplace=True)
    try:
        pp.runpp(net_temp, init="auto")
    except:
        critical_lines.append(i)

# Report critical lines that break the system (i.e., not N-1 secure)
critical_lines_ids = net.line.loc[critical_lines, ['from_bus', 'to_bus']]

# Add synthetic PMU measurements (voltage magnitude and angle) to the system
# For simplicity, let's simulate PMUs at buses 0 to 9
pmu_buses = list(range(10))
std_vm = 0.01  # standard deviation for voltage magnitude measurements
std_va = 0.01  # standard deviation for voltage angle measurements

for bus in pmu_buses:
    vm_value = net.res_bus.vm_pu.at[bus]
    va_value = net.res_bus.va_degree.at[bus]
    create.create_measurement(net, "v", "bus", vm_value, std_vm, bus)
    create.create_measurement(net, "va", "bus", va_value, std_va, bus)

# Create FDIA (False Data Injection Attack) by altering some measurement values maliciously
# For example, we corrupt voltage magnitude measurements at buses 5 and 7
for meas in net.measurement.itertuples():
    if meas.element in [5, 7] and meas.measurement_type == "v":
        net.measurement.at[meas.Index, "value"] += 0.1  # inject a 10% false reading

# Run state estimation
estimation_result = est.estimate(net)

# Gather estimation results (voltage magnitude and angle)
estimated_vm = net.res_bus_est.vm_pu
estimated_va = net.res_bus_est.va_degree

import pandas as pd
import ace_tools as tools

# Return important results for display
df_result = pd.DataFrame({
    "Bus": net.bus.index,
    "Estimated VM (p.u.)": estimated_vm,
    "Estimated VA (deg)": estimated_va
})

tools.display_dataframe_to_user(name="Estimated State with FDIA", dataframe=df_result)

# Output critical lines to user
critical_lines_ids.reset_index(drop=True, inplace=True)
tools.display_dataframe_to_user(name="Critical Lines (Not N-1 Secure)", dataframe=critical_lines_ids)
