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
# position
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
        self.channels = configs.enc_in
        self.individual = configs.individual
        self.hidden_size = configs.hidden_size
        # if self.individual:
        #     self.Linear = nn.ModuleList()
        #     for i in range(self.channels):
        #         self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        # else:
        self.Linear_encoder = nn.Linear(self.channels, self.hidden_size)
        self.Lstm = nn.LSTM(self.hidden_size, self.hidden_size,batch_first=True) 
        self.Linear_decoder = nn.Linear(self.hidden_size,self.channels)
        # self.Linear4 = nn.Linear(256,128)
        # self.Linear3 = nn.Linear(41,41)
        # self.tanh=nn.Tanh()

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # print(x.shape)
        x=self.Linear_encoder(x)
        # x=self.relu(x)
        
        # print(x.shape)
        outputs,(h,c)=self.Lstm(x)  #output(batch,len,dim)  x(batch,1,dim) c(batch,1,dim)
        # print('1',outputs.shape)
        for i in range(self.pred_len):
            output,(h0,c)=self.Lstm(h.permute(1,0,2))
            outputs=torch.cat((outputs,h0.permute(1,0,2)),1)
            h=h0
        # print('2',outputs.shape)
        # x=x[0]
        # print(x.shape)
        
        x=outputs[:,1:,:]
        # x=self.Linear4(x)
        x=self.Linear_decoder(x)
        # x=self.Linear3(x.permute(0,2,1)).permute(0,2,1)
        # x=self.relu(x)
        # print(x.shape)
        # x=F.normalize(x, p=2, dim=2, eps=1e-12, out=None)
        return x # [Batch, Output length, Channel]






# # position
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
#         self.hidden_size = configs.hidden_size
#         # if self.individual:
#         #     self.Linear = nn.ModuleList()
#         #     for i in range(self.channels):
#         #         self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
#         # else:
#         self.Linear_encoder = nn.Linear(self.channels, self.hidden_size)
#         self.Lstm = nn.LSTM(self.hidden_size, self.hidden_size,batch_first=True) 
#         self.Linear_decoder = nn.Linear(self.hidden_size,self.channels)
#         # self.Linear4 = nn.Linear(256,128)
#         # self.Linear3 = nn.Linear(41,41)
#         # self.tanh=nn.Tanh()

#     def forward(self, x):
#         # x: [Batch, Input length, Channel]
#         # print(x.shape)
#         x=self.Linear_encoder(x)
#         # x=self.relu(x)
        
#         # print(x.shape)
#         outputs,(h,c)=self.Lstm(x)  #output(batch,len,dim)  x(batch,1,dim) c(batch,1,dim)
#         # print('1',outputs.shape)
#         for i in range(self.pred_len):
#             output,(h0,c)=self.Lstm(h.permute(1,0,2))
#             outputs=torch.cat((outputs,h0.permute(1,0,2)),1)
#             h=h0
#         # print('2',outputs.shape)
#         # x=x[0]
#         # print(x.shape)
#         x=outputs[:,1:,:]
#         # x=self.Linear4(x)
#         x=self.Linear_decoder(x)
#         # x=self.Linear3(x.permute(0,2,1)).permute(0,2,1)
#         # x=self.relu(x)
#         # print(x.shape)

#         return x # [Batch, Output length, Channel]
    
    