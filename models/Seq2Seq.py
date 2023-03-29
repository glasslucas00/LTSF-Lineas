import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, hidden = self.lstm(x)
        return hidden

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        output = self.linear(output)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.decoder = Decoder(input_dim, hidden_dim, output_dim, num_layers)

    def forward(self, x, y):
        encoder_hidden = self.encoder(x)
        decoder_output, _ = self.decoder(y, encoder_hidden)
        return decoder_output

input_dim = 10
hidden_dim = 20
output_dim = 1
num_layers = 2

model = Seq2Seq(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)