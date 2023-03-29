# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d

np.set_printoptions(threshold=np.inf)
file_dir='NMP'
save_dir='Dataset'
    # %%
def quatProduct(q1, q2):
    r1 = q1[3]
    r2 = q2[3]
    v1 = np.array([q1[0], q1[1], q1[2]])
    v2 = np.array([-q2[0], -q2[1], -q2[2]])

    r = r1 * r2 - np.dot(v1, v2)
    v = r1 * v2 + r2 * v1 + np.cross(v1, v2)
    q = np.array([ v[0], v[1], v[2],r])
    return q
def rotate(position, quaternion):
    # fig = plt.figure(figsize=(10,10))
    # #创建3D坐标系
    # ax = fig.gca(projection='3d')
    # ax.set_xlim(-2.5,2.5)
    # ax.set_ylim(-3,3)
    # ax.set_zlim(0,2.5)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')

    angle=np.random.uniform(-np.pi, np.pi)
    r = R.from_rotvec([0, 0,angle])
    angle=angle*180/np.pi
    # print(angle)
    position2=r.apply(position)
    data1=position
    data2=position2
    quaternion2=quaternion[:]
    for q in quaternion2:
        q=quatProduct(q, r.as_quat())
    # ax.plot(data1[:,0], data1[:,1],data1[:,2], label='0_xyz',c='r')
    # ax.plot(data2[:,0], data2[:,1],data2[:,2], label='xyz',c='g')
    # for i in range(data1.shape[0]-1):
    #     # ax.quiver(data1[i][0],data1[i][1],data1[i][2],data1[i][4],data1[i][5],data1[i][6],length=data1[i][3], arrow_length_ratio=0.1,color='b')
    #     ax.quiver(data1[i][0],data1[i][1],data1[i][2],quaternion[i][0],quaternion[i][1],quaternion[i][2],length=quaternion[i][3], arrow_length_ratio=0.1,color='y')

    # for i in range(data2.shape[0]-1):
    #     # ax.quiver(data2[i][0],data2[i][1],data2[i][2],data2[i][4],data2[i][5],data2[i][6],length=data2[i][3], arrow_length_ratio=0.1,color='b')
    #     ax.quiver(data2[i][0],data2[i][1],data2[i][2],quaternion2[i][0],quaternion2[i][1],quaternion2[i][2],length=quaternion2[i][3], arrow_length_ratio=0.1,color='gray')
    # plt.show()
    return position2,quaternion2,angle
def data_process(data,rotation_flag):
    # data = np.load('./NMP/trimmed_Bottle/trimmed_Bottle_53.npz')
    #order
    #times
    time_step=data['time_step']
    time_step=time_step-time_step[0]
    # time_step = time_step[:,np.newaxis]
    position=data['position'][:,[0,2,1]]
    quaternion=data['quaternion']
    if rotation_flag:
        position, quaternion,angle=rotate(position, quaternion)
    else:
        angle=0


    # data_np=np.concatenate((time_step,position),axis=1)
    # data_np=np.concatenate((time_step,position,quaternion),axis=1)

    # print(data_np.shape)


    # %%
    # 计算位移矩阵
    D = np.diff(position, axis=0)
    
    Ts=time_step
    # 计算时间间隔向量
    T = np.diff(Ts)

    # 计算速度矩阵
    V = D / T.reshape((-1, 1))

    # 计算加速度矩阵
    A = np.diff(V, axis=0) / (T[1:].reshape((-1, 1)) + T[:-1].reshape((-1, 1)))

    

    # data_np=np.concatenate((data_np[1:-1],V,A),axis=1)
    # print(data_np.shape)

    Q=[]
    for i in range(quaternion.shape[0]-1):
        s=quatProduct(quaternion[i+1],quaternion[i])
        Q.append(s)
    Q=np.array(Q)
    # print(D.shape,T.shape,V.shape,A.shape,quaternion.shape,Q.shape)

    time_step = time_step[:,np.newaxis]

    data_np=np.concatenate((time_step[1:-1],position[1:-1],V[:-1],A,quaternion[1:-1],Q[:-1]),axis=1)
    # %%
    # print(Q.shape)
    # data_np=np.concatenate((data_np,quaternion[1:-1],Q),axis=1)
    for i in range(data_np.shape[1]):
        data_np[:,i]=gaussian_filter1d(data_np[:,i],2)
    
# print(data_np.shape)
    # %%
    # fig = plt.figure(figsize=(10,10))
    # #创建3D坐标系
    # ax = fig.gca(projection='3d')
    # ax.set_xlim(-2.5,2.5)
    # ax.set_ylim(-3,3)
    # ax.set_zlim(0,2.5)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # # data2=data_np[:,[1,2,3,-4,-3,-2,-1]]
    # data2=data_np[:,[1,2,3,-8,-7,-6,-5,4,5,6,7,8,9]]
    # ax.quiver(data2[0][0],data2[0][1],data2[0][2],data2[20][0]-data2[0][0],data2[20][1]-data2[0][0],data2[20][2]-data2[0][0],length=1, arrow_length_ratio=0.1,color='y')
    # for i in range(data2.shape[0]-1):
    #     # ax.quiver(data2[i][0],data2[i][1],data2[i][2],data2[i][4],data2[i][5],data2[i][6],length=data2[i][3], arrow_length_ratio=0.1,color='b')
    #     ax.quiver(data2[i][0],data2[i][1],data2[i][2],data2[i][3],data2[i][4],data2[i][5],length=data2[i][6], arrow_length_ratio=0.1,color='g')
    #     ax.quiver(data2[i][0],data2[i][1],data2[i][2],data2[i][7],data2[i][8],data2[i][9],length=0.4, arrow_length_ratio=0.1,color='r')
    #     ax.quiver(data2[i][0],data2[i][1],data2[i][2],data2[i][10],data2[i][11],data2[i][12],length=0.3, arrow_length_ratio=0.1,color='b')
    # data2=data_np[:,1:4].transpose()
    # ax.plot(data2[0], data2[1],data2[2], label='xyz',c='r')
    # plt.show()

    return data_np,angle
# %%

data = np.load('./NMP/trimmed_Bottle/trimmed_Bottle_53.npz')
s,a=data_process(data,0)
print(a)