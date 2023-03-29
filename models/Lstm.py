
import torch
from torch import nn
class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(LSTMClassifier, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) 
        self.lstm = nn.LSTM(embedding_length, hidden_size) # Our main hero for this tutorial
        self.label = nn.Linear(hidden_size, output_size)
    def forward(self, input_sentence, batch_size=None):
        input = self.word_embeddings(input_sentence) 
        input = input.permute(1, 0, 2) 

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_output = self.label(final_hidden_state[-1]) 
        
        return final_output
    
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
        # if self.individual:
        #     self.Linear = nn.ModuleList()
        #     for i in range(self.channels):
        #         self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        # else:
        self.Linear1 = nn.Linear(self.channels, 256)
        self.Lstm = nn.LSTM(256, 256,batch_first=True) 
        self.Linear2 = nn.Linear(128,self.channels)
        self.Linear4 = nn.Linear(256,128)
        self.Linear3 = nn.Linear(41,41)
        self.tanh=nn.Tanh()

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # print(x.shape)
        x=self.Linear1(x)
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
        x=self.Linear4(outputs[:,1:,:])
        x=self.Linear2(x)
        # x=self.Linear3(x.permute(0,2,1)).permute(0,2,1)
        # x=self.relu(x)
        # print(x.shape)

        return x # [Batch, Output length, Channel]