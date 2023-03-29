import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class Model(nn.Module):
#     """
#     Just one Linear layer
#     """
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
        
#         # Use this line if you want to visualize the weights
#         # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#         self.channels = configs.enc_in
#         self.individual = configs.individual
#         if self.individual:
#             self.Linear = nn.ModuleList()
#             for i in range(self.channels):
#                 self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
#         else:
#             self.Linear = nn.Linear(self.seq_len, self.pred_len)

#     def forward(self, x):
#         # x: [Batch, Input length, Channel]
#         if self.individual:
#             output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
#             for i in range(self.channels):
#                 output[:,:,i] = self.Linear[i](x[:,:,i])
#             x = output
#         else:
#             x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
#         return x # [Batch, Output length, Channel]
class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        if configs.enc_in==7:
            self.channels1 = 3
            self.channels2 = 4
        self.individual = configs.individual
        self.hidden_size = configs.hidden_size
        # if self.individual:
        #     self.Linear = nn.ModuleList()
        #     for i in range(self.channels):
        #         self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        # else:
        self.Linear_encoder = nn.Linear(self.channels1, self.hidden_size)
        self.Quat_encoder = nn.Linear(self.channels2, self.hidden_size)
        self.Linear_lstm = nn.LSTM(self.hidden_size, self.hidden_size,batch_first=True) 
        self.Quat_lstm = nn.LSTM(self.hidden_size, self.hidden_size,batch_first=True) 
        self.Linear_decoder = nn.Linear(self.hidden_size,self.channels1)
        self.Quat_decoder = nn.Linear(self.hidden_size,self.channels2)
        # self.Linear4 = nn.Linear(256,128)
        # self.Linear3 = nn.Linear(41,41)
        # self.tanh=nn.Tanh()


    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # print(x.shape)
        lx=self.Linear_encoder(x[:,:,:3])
        rx=self.Quat_encoder(x[:,:,3:])
        # x=self.relu(x)
        
        # print(x.shape)
        outputs,(h,c)=self.Linear_lstm(lx)  #output(batch,len,dim)  x(batch,1,dim) c(batch,1,dim)
        # print('1',outputs.shape)
        for i in range(self.pred_len):
            output,(h0,c)=self.Linear_lstm(h.permute(1,0,2))
            outputs=torch.cat((outputs,h0.permute(1,0,2)),1)
            h=h0
        lx=outputs[:,1:,:]

        outputs,(h,c)=self.Quat_lstm(rx)  #output(batch,len,dim)  x(batch,1,dim) c(batch,1,dim)
        # print('1',outputs.shape)
        for i in range(self.pred_len):
            output,(h0,c)=self.Quat_lstm(h.permute(1,0,2))
            outputs=torch.cat((outputs,h0.permute(1,0,2)),1)
            h=h0 
        rx=outputs[:,1:,:]    
        # x=self.Linear4(x)
        lx=self.Linear_decoder(lx)
        rx=self.Quat_decoder(rx)
        x=torch.cat((lx,rx),2)
        # x=self.Linear3(x.permute(0,2,1)).permute(0,2,1)
        # x=self.relu(x)
        # print(x.shape)

        return x # [Batch, Output length, Channel]