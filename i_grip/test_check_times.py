import os
import numpy as np
#read file ./monitoring_queue_NOexec.txt
times = []
with open('/home/emoullet/tqst_no.txt', 'r') as f:
    #get lines with 'mean check all targets time'
    lines = f.readlines()
    #print(lines)
    #print(len(lines))
    for line in lines:
        if 'mean_check_all_targets_time' in line:
            time= line.split()[-1]
            times.append(float(time))
        
print(np.mean(times))
times = []
a = 0
i=0
with open('/home/emoullet/tqst2.txt', 'r') as f:
    #get lines with 'mean check all targets time'
    lines = f.readlines()
    # print(lines)
    #print(len(lines))
    for line in lines:
        if 'mean_check_all_targets_time' in line:
            time= line.split()[-1]
            a=float(time)
            # test if a is nan
            if not np.isnan(a):
                times.append(float(time))
# print(times)
print(np.mean(times))