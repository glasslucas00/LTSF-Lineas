import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
from utils.tools import StandardScaler
from utils.timefeatures import time_features
from scipy.spatial.transform import Rotation as R
import warnings
warnings.filterwarnings('ignore')
def scale_quaternion(q, k):
    # 计算旋转角度的余弦值
    cos_theta = q[3]
    # 计算旋转角度的缩放系数
    scale = np.sqrt(1 - k**2 + k**2 * cos_theta**2)
    # 构造新的四元数
    q_new = np.array([ q[0]*k, q[1]*k, q[2]*k,scale])
    return q_new
def quatProduct(q1, q2):
    r1 = q1[3]
    r2 = q2[3]
    v1 = np.array([q1[0], q1[1], q1[2]])
    v2 = np.array([-q2[0], -q2[1], -q2[2]])

    r = r1 * r2 - np.dot(v1, v2)
    v = r1 * v2 + r2 * v1 + np.cross(v1, v2)
    q = np.array([ v[0], v[1], v[2],r])
    return q
def quatMutli(q1, q2):
    r1 = q1[3]
    r2 = q2[3]
    v1 = np.array([q1[0], q1[1], q1[2]])
    v2 = np.array([q2[0], q2[1], q2[2]])

    r = r1 * r2 - np.dot(v1, v2)
    v = r1 * v2 + r2 * v1 + np.cross(v1, v2)
    q = np.array([ v[0], v[1], v[2],r])
    return q
def rotate_xyz (data_xyz):
    #轨迹从原点开始
    data_xyz=data_xyz-data_xyz[0]
    si=2
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
def rotate_data (df_data,num):
    # df_data[:,4:]=rotate_xyz(df_data[:,4:])
    datas=[]
    for i in range(num):
        randi=np.random.uniform(-179.9,179.9)
        xyz_angle=np.pi*randi/180
        r = R.from_rotvec([0, 0,xyz_angle])
        # for quat in df_data[:,:4]:
        #     quat=quatMutli(quat,scale_quaternion(R.random().as_quat(), np.random.uniform(-0.05,0.05)))
        df_data[:,:3]=r.apply(df_data[:,:3])
        # trans_q=np.random.uniform(-1,1,(3))
        # df_data[:,4:]+=trans_q
        datas.append(df_data)

    # for i in range(num):
    #     xyz_angle=np.pi*i*-1/num
    #     r = R.from_rotvec([0, 0, xyz_angle])
    #     for quat in df_data[:,:4]:
    #         quat=quatMutli(quat,scale_quaternion(R.random().as_quat(), np.random.uniform(-0.05,0.05)))
    #     df_data[:,4:]=r.apply(df_data[:,4:])
    #     datas.append(df_data)
    return datas
# def rotate_data (data_xyz,num):
#     data_xyz=rotate_xyz(data_xyz)
#     datas=[]
#     for i in range(num):
#         xyz_angle=np.pi*i/num
#         r = R.from_rotvec([0, 0,xyz_angle])
#         trans_xyz=r.apply(data_xyz)
#         datas.append(trans_xyz)
#     for i in range(num):
#         xyz_angle=np.pi*i*-1/num
#         r = R.from_rotvec([0, 0, xyz_angle])
#         trans_xyz=r.apply(data_xyz)
#         datas.append(trans_xyz)
#     return datas





    
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
            if  f :
                # if 'Bamboo' in f:
                df_raw = pd.read_csv(os.path.join(self.root_path,f))
                if self.features=='M' or self.features=='MS':
                    # cols_data = df_raw.columns[5:]
                    # df_data = df_raw[cols_data]
                    # df_data =df_raw[['x', 'y', 'z']].values
                    # df_data =df_raw[['di', 'dj', 'dk','dw']].values
                    # df_data =df_raw[['i', 'j', 'k','w']].values
                    df_data =df_raw[['x', 'y', 'z','i', 'j', 'k','w']].values
                    # df_data =df_raw[['x', 'y', 'z','vx', 'vy', 'vz','ax', 'ay', 'az','i', 'j', 'k','w','di', 'dj', 'dk','dw']].values
                elif self.features=='S':
                    df_data = df_raw[[self.target]]
                # print(np.array(df_data).shape)

                # #tran quat
                # quats=df_data[:,:4]
                # si=2
                # quats[:,0] = gaussian_filter1d(quats[:,0], sigma=si)
                # quats[:,1] = gaussian_filter1d(quats[:,1], sigma=si)
                # quats[:,2] = gaussian_filter1d(quats[:,2], sigma=si)
                # quats[:,3] = gaussian_filter1d(quats[:,3], sigma=si)
                # for j in range(quats.shape[0]-1):
                #     #画起点为(0,0,0),终点为(1,1,1)的向量
                #     quats[j]= quatProduct(quats[j+1],quats[j])
                # quats[0]=quats[1]
                # df_data[:,:4]=quats

        
                # datas= rotate_data(df_data,30)
                # df_data[:,4:]=xyz_datas
                # print(df_data.shape)
                # df_data[:,:3]=df_data[:,:3]-df_data[0,:3]
                data_stamp = df_raw[['date']][:]
                data_stamp=np.array(data_stamp)
                # for df_data in datas:
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
        # print(y[5]) 
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