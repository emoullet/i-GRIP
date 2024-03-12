import matplotlib.pyplot as plt

# open file log.txt
# with open('log.txt', 'r') as file:
#     # read file and find keywords for every line containing 'ms'
#     key_words = []
#     for line in file:
#         if 'ms' in line:
#             line_str = line.split()
#             keyword_split = line_str[:-2]
#             # delete ':' if present
#             for i in range(len(keyword_split)):
#                 if ':' in keyword_split[i]:
#                     keyword_split[i] = keyword_split[i][:-1]
#             keyword = ' '.join(keyword_split)
#             if keyword not in key_words:
#                 key_words.append(keyword)
#             # break
#     print(key_words)

keys = ['frame collection time ', 'polling time ', 'sending time ', 'frame sending time ', 'get hands and objects time ', 'hand render time ', 'obj render time ', 'scene render', 'updated img ', 'scene update hands', 'predict_future_trajectory time ', 'update_hands_meshes time ', 'update_object_meshes time ', 'update_target_detectors_meshes time ', 'copy hands time ', 'update_trajectory_meshes time ', 'check_all_targets time ', 'fetch_all_targets time ', 'update_meshes time ', 'draw_landmarks tie ', 'check_all_targets time for hand left ', 'analyse all targets', 'mean_check_all_targets_time', 'get most probable target', 'detect_hands_task', 'update_object_meshes time ', 'predict_future_trajectory time ', 'plotter time receive data', 'propagate all hand time']
# keys = ['predict_future_trajectory time ', 'get_future_trajectory time ', 'delete_geometry time', 'add_geometry time']
# delete duplicates
keys = ['check_all_targets time ']
keys = list(set(keys))

data_x = {}
data_y = {}
for key in keys:
    data_x[key] = []
    data_y[key] = []
file_path = 'log.txt'
file_path = 'log_thread.txt'
with open(file_path, 'r') as file:
    # read file and find keywords for every line containing 'ms'
    key_words = []
    for line in file:
        for key in keys:
            if key in line :
                print(line)
                split_line = line.split()
                if len(split_line) < 2:
                    continue
                if split_line[-1] != 'ms':
                    continue
                time = split_line[-2]
                data_y[key].append(float(time))
                data_x[key].append(len(data_y[key]))
                break
print(data_x)
print(data_y)
# plot data
fig, ax = plt.subplots()
for key in keys:
    ax.plot(data_x[key], data_y[key], label=key)
plt.legend()
plt.show()

for key in keys:
    print(f'{key} data: {data_y[key]}')
    if len(data_y[key]) > 0:
        data_y[key] = data_y[key][500:]
        data_x[key] = data_x[key][500:]
        print(f'{key} mean: {sum(data_y[key])/len(data_y[key])} ms')
        print(f'{key} std: {sum([(x - sum(data_y[key])/len(data_y[key]))**2 for x in data_y[key]])/len(data_y[key])} ms')
    fig, ax = plt.subplots()
    ax.plot(data_x[key], data_y[key], label=key)
    plt.legend()
    plt.show()