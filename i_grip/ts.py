import pandas as pd

path = '/home/emoullet/Documents/i-GRIP/DATA/Session_1_processing/Pre_processing/RKS6625/trial_2_combi_mustard_left_pinch_simulated/trial_2_combi_mustard_left_pinch_simulated_cam_1944301011EA1F1300_timestamps_movement.gzip'
path = '/home/emoullet/Documents/i-GRIP/DATA/Session_1_processing/Pre_processing/RKS6625/trial_2_combi_mustard_left_pinch_simulated/trial_2_combi_mustard_left_pinch_simulated_cam_1944301011EA1F1300_depth_map_movement.gzip'
table = pd.read_pickle(path, compression='gzip')
print(table)

path = '/home/emoullet/Documents/i-GRIP/DATA/Session_1_processing/Replay/RKS6625/trial_4_combi_cheez\'it_right_palmar_simulated/trial_4_combi_cheez\'it_right_palmar_simulated_cam_19443010910F481300_main.csv'
table = pd.read_csv(path)
print(table)
# delete duplicate rows
table = table.drop_duplicates(subset=['Timestamps'])
print(table)
print(table.columns)