import pandas as pd

"""
The function is to get solar power output of selected pv farms from the file named "output_solar"
"""
# df = pd.read_csv('zip\\output_solar.csv')
#
# resource_names = [
#     "SAMSON_1_G1",
#     "SAMSON_3_G1",
#     "KELAM_SL_UNIT1",
#     "GOLINDA_UNIT1",
#     "PISGAH_SOLAR1",
#     "BLUEJAY_UNIT1",
#     "ELARA_SL_UNIT1",
#     "RADN_SLR_UNIT1",
#     "FWLR_SLR_UNIT1"
# ]
#
# # Filter the DataFrame
# df_slice = df[df["Resource Name"].isin(resource_names)]
# columns_to_select = [
#     "SCED Time Stamp",
#     "QSE",
#     "DME",
#     "Resource Name",
#     "Resource Type",
#     "SCED1 Curve-MW3",
#     "Output Schedule",
#     "HSL",
#     "HASL",
#     "HDL",
#     "Base Point",
#     "Telemetered Net Output "    ### this column is for real output
# ]
# print(df_slice)
#
# # Select specific columns from df_slice
# df_slice_selected = df_slice[columns_to_select]
#
# df_slice_selected.set_index("SCED Time Stamp", inplace=True)
# df_slice_selected.to_csv("chosen_pv.csv")
# print(df_slice_selected)