import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
# from utils.tools import StandardScaler
# from utils.timefeatures import time_features
from scipy.spatial.transform import Rotation as R
import warnings
warnings.filterwarnings('ignore')
# xyz

def rotate_xyz (data_xyz):
    #轨迹从原点开始
    data_xyz=data_xyz-data_xyz[0]+np.array([0,0,1])
    si=5
    #滤波
    data_xyz[:,0] = gaussian_filter1d(data_xyz[:,0], sigma=si)
    data_xyz[:,1] = gaussian_filter1d(data_xyz[:,1], sigma=si)
    data_xyz[:,2] = gaussian_filter1d(data_xyz[:,2], sigma=si)
    #绕z旋转
    a=np.array([data_xyz[1,0],data_xyz[1,1]])
    b=np.array([1,0])
    cos_ = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    sin_ = np.cross(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    arctan2_ = np.arctan2(sin_, cos_)
    r = R.from_rotvec([0, 0, arctan2_])
    trans_xyz=r.apply(data_xyz)
    return trans_xyz
def rotate_data (data_xyz,num):
    data_xyz=rotate_xyz(data_xyz)
    datas=[]
    for i in range(num):
        angle=np.pi*i/num
        r = R.from_rotvec([0, 0,angle])
        trans_xyz=r.apply(data_xyz)
        datas.append(trans_xyz)
    for i in range(num):
        angle=np.pi*i*-1/num
        r = R.from_rotvec([0, 0, angle])
        trans_xyz=r.apply(data_xyz)
        datas.append(trans_xyz)
    return datas
class Dataset_Traject(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        # self.__getitem__(1)

    def __read_data__(self):
        files=os.listdir(self.root_path)
        x=[]
        y=[]
        x_mark=[]
        y_mark=[]
        for f in files[:]:
            df_raw = pd.read_csv(os.path.join(self.root_path,f))
            df_data =df_raw[['x','z','y']].values
            data_stamp = df_raw[['date']][:]
            data_stamp=np.array(data_stamp)
            datas= rotate_data(df_data,10)
            for df_data in datas:


            # print(df_data.shape)

                # df_stamp['date'] = pd.to_datetime(df_stamp.date,unit='ns')  
                # data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
                # print(data_stamp[:5,:])
                for i in range((df_data.shape[0])-self.seq_len -self.pred_len):
                    s_begin = i
                    s_end = s_begin + self.seq_len
                    r_begin = s_end - self.label_len 
                    r_end = r_begin + self.label_len + self.pred_len
                    _x = df_data[s_begin:s_end,:] ## 16 columns for features  
                    _y = df_data[r_begin:r_end,:] ## column 0 contains the labbel
                    _x_mark=data_stamp[s_begin:s_end,:] 
                    _y_mark=data_stamp[r_begin:r_end,:] 
                    x.append(_x)
                    y.append(_y)
                    x_mark.append(_x_mark)
                    y_mark.append(_y_mark)
                

        print(np.array(x).shape,np.array(y).shape,np.array(x_mark).shape,np.array(y_mark).shape)
        self.data_x=np.array(x)    
        self.data_y=np.array(y)
        self.data_x_mark=np.array(x_mark)  
        self.data_y_mark=np.array(y_mark)  
        self.len = len(x) 

    
    def __getitem__(self, index):
        seq_x, seq_y, seq_x_mark, seq_y_mark=self.data_x[index],self.data_y[index],self.data_x_mark[index],self.data_y_mark[index]
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return  self.len


    def inverse_transform(self, data):
        return data
    

def data_provider(args, flag):
    global train_loader,test_loader,val_loader,train_dataset, test_dataset,val_dataset
    # timeenc = 0
    # shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
    # data_set = Dataset_Traject(
    #     root_path=args.root_path,
    #     data_path=args.data_path,
    #     flag='train',
    #     size=[args.seq_len, args.label_len, args.pred_len],
    #     features=args.features,
    #     target=args.target,
    #     timeenc=timeenc,
    #     freq='m',
    # )
    # train_size = int(len(data_set) * 0.8)
    # test_size = int(len(data_set) * 0.1)
    # val_size =len(data_set)-train_size-test_size
    # train_dataset, test_dataset,val_dataset = torch.utils.data.random_split(data_set, [train_size, test_size,val_size])
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     drop_last=False
    #     )
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     drop_last=True
    #     )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     drop_last=True
    #     )
    if flag == 'test':
        return test_dataset,test_loader
    elif flag == 'pred':
        return val_dataset,val_loader
    else:
        return train_dataset,train_loader 
def data_make(args):
    global train_loader,test_loader,val_loader,train_dataset, test_dataset,val_dataset
    timeenc = 0
    shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
    data_set = Dataset_Traject(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='train',
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq='m',
    )
    train_size = int(len(data_set) * 0.8)
    test_size = int(len(data_set) * 0.1)
    val_size =len(data_set)-train_size-test_size
    train_dataset, test_dataset,val_dataset = torch.utils.data.random_split(data_set, [train_size, test_size,val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True
        )