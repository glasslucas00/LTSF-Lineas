import numpy as np
import pandas as pd
import os
from data_p import data_process
file_dir='../dataset/NMP'
save_dir='../dataset/Dataset2'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for f in os.listdir(file_dir):
    for file in os.listdir(os.path.join(file_dir,f)):
        data = np.load(os.path.join(os.path.join(file_dir, f),file))
        data_np,angle=data_process(data,1)
        #order
        
        # frame_num=data['frame_num']
        # frame_num=np.arange(frame_num.shape[0])

        # #times
        # time_step=data['time_step']
        # time_step=time_step-time_step[0]
        # frame_num = frame_num[:,np.newaxis]
        # time_step = time_step[:,np.newaxis]

        # position=data['position']
        # quaternion=data['quaternion']

        # data_np=np.concatenate((time_step,position,quaternion),axis=1)
        # print(data_np.shape)


        df1 = pd.DataFrame(data=data_np,columns=['date' ,'x', 'y', 'z','vx', 'vy', 'vz','ax', 'ay', 'az','i', 'j', 'k','w','di', 'dj', 'dk','dw'])
        filename=file.split('.')[0]+'_'+str(angle)+'.csv'
        df1.to_csv(os.path.join(save_dir,filename) ,index=False)
# np.concatenate()
# for key, value in data.items():
#     # np.concatenate()
#     print(key,value.shape)
    # np.savetxt("aa" + key + ".csv", value)