import pandas as pd
import numpy as np
import pandapower as pp
from create_net import create_network  # Ensure this is correct
import os
import warnings

warnings.filterwarnings("ignore")

# Load time-series generation and load data
gen = pd.read_csv(r"C:\Users\jliu359\Downloads\Resilience\gen_118.csv", delimiter=',')
load = pd.read_csv(r"C:\Users\jliu359\Downloads\Resilience\load_118.csv", delimiter=',')

# Set time index
gen = gen.set_index('SCED Time Stamp')
load = load.set_index('Hour Ending')

# Generator adjustments
gen['5'] *= 3
gen['25'] *= 3
gen['64'] *= 2
gen['69'] *= 2
gen['3'] = 200
gen['53'] = 300
gen['76'] = 300

# Rename columns
gen.columns = [f'gen{i}' for i in range(gen.shape[1])]
load.columns = [f'load{i}' for i in range(load.shape[1])]

# Prepare output directory
output_dir = "n1_results_by_hour"
os.makedirs(output_dir, exist_ok=True)

# Store non-converged cases
nonconverged_cases = []

# Loop over all timestamps
for timestamp in gen.index:
    print(f"\n⏳ Running N-1 analysis for time: {timestamp}")

    # Extract load and generation for this timestep
    load_data = load.loc[[timestamp]]
    gen_data = gen.loc[[timestamp]]

    # Build the network
    net = create_network(load_data, gen_data)
    results = []

    # Loop through each line for N-1
    for line_idx in net.line.index:
        net_copy = net.deepcopy()
        net_copy.line.at[line_idx, 'in_service'] = False

        try:
            pp.runpp(net_copy)
            success = True
            max_loading = net_copy.res_line.loading_percent.max()
            min_vm = net_copy.res_bus.vm_pu.min()
            max_vm = net_copy.res_bus.vm_pu.max()

            print(f"✅ Line {line_idx} out: Converged | Max Loading: {max_loading:.2f}% | Vm range: {min_vm:.3f}–{max_vm:.3f}")
        except pp.LoadflowNotConverged:
            success = False
            max_loading = None
            min_vm = None
            max_vm = None

            print(f"❌ Line {line_idx} out: Did NOT converge")
            nonconverged_cases.append({
                "timestamp": timestamp,
                "line_idx": line_idx,
                "from_bus": net.line.at[line_idx, 'from_bus'],
                "to_bus": net.line.at[line_idx, 'to_bus']
            })

        results.append({
            "timestamp": timestamp,
            "line_idx": line_idx,
            "from_bus": net.line.at[line_idx, 'from_bus'],
            "to_bus": net.line.at[line_idx, 'to_bus'],
            "success": success,
            "max_loading_percent": max_loading,
            "min_vm_pu": min_vm,
            "max_vm_pu": max_vm
        })

    # Save N-1 result for current timestep
    safe_timestamp = timestamp.replace(":", "-").replace(" ", "_")
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, f"n1_result_{safe_timestamp}.csv"), index=False)

# Save non-converged cases summary
if nonconverged_cases:
    df_fail = pd.DataFrame(nonconverged_cases)
    df_fail.to_csv(os.path.join(output_dir, "nonconverged_cases.csv"), index=False)
    print(f"\n⚠️ Total non-converged cases: {len(nonconverged_cases)}. See 'nonconverged_cases.csv' in {output_dir}")
else:
    print("\n✅ All line outages converged successfully across all timesteps!")

print("\n🏁 N-1 time-varying analysis completed. All results stored in 'n1_results_by_hour'")
