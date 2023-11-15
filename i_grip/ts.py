import pandas as pd

path = '/home/emoullet/Documents/i-GRIP/DATA/Session_1_processing/Pre_processing/RKS6625/trial_2_combi_mustard_left_pinch_simulated/trial_2_combi_mustard_left_pinch_simulated_cam_1944301011EA1F1300_timestamps_movement.gzip'
path = '/home/emoullet/Documents/i-GRIP/DATA/Session_1_processing/Pre_processing/RKS6625/trial_2_combi_mustard_left_pinch_simulated/trial_2_combi_mustard_left_pinch_simulated_cam_1944301011EA1F1300_depth_map_movement.gzip'
table = pd.read_pickle(path, compression='gzip')
print(table)