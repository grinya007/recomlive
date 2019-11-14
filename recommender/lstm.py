import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTM(nn.Module):
    def __init__(
            self,
            input_size,
            embedding_size,
            hidden_size,
            device = torch.device('cuda'),
            lr = 0.001
    ):
        super(LSTM, self).__init__()
        self.device      = device
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(input_size, embedding_size)
        self.lstm        = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.dense       = nn.Linear(hidden_size, input_size)
        self.criterion   = nn.CrossEntropyLoss()
        self.optimizer   = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)
    
    def forward(self, x, h):
        embed     = self.embedding(x)
        output, h = self.lstm(embed, h)
        logits    = self.dense(output)
        return logits, h
    
    def zero_state(self, batch_size):
        return (
            torch.zeros(1, batch_size, self.hidden_size).to(self.device),
            torch.zeros(1, batch_size, self.hidden_size).to(self.device),
        )
    
    def fit(self, x, h = None):
        batch_size = x.shape[0] - 1
        if isinstance(h, type(None)):
            h = self.zero_state(batch_size)
        elif (h[0].shape[1] > batch_size):
            h = (
                h[0][:, -batch_size:],
                h[1][:, -batch_size:],
            )
        elif (h[0].shape[1] < batch_size):
            h_zero = self.zero_state(batch_size - h[0].shape[1])
            h = (
                torch.cat((h[0], h_zero[0]), dim=1),
                torch.cat((h[1], h_zero[1]), dim=1),
            )
        
        y = x[1:]
        x = x[:-1]
        
        self.train()
        self.optimizer.zero_grad()
        x = torch.tensor(x).to(self.device)
        y = torch.tensor(y).to(self.device)
        
        logits, (state_h, state_c) = self.forward(x, h)
        loss = self.criterion(logits.transpose(1, 2), y)

        state_h = state_h.detach()
        state_c = state_c.detach()

        loss_value = loss.item()

        loss.backward()

        nn.utils.clip_grad_norm_(self.parameters(), 5)

        self.optimizer.step()
        
        return loss_value, (state_h, state_c)
    
    def predict(self, x, h = None):
        if isinstance(h, type(None)):
            h = self.zero_state(1)
        
        x = torch.tensor(x).to(self.device)
        logits, (state_h, state_c) = self.forward(x, (h[0][:, -1:], h[1][:, -1:]))
        state_h = state_h.detach()
        state_c = state_c.detach()
        
        _, prediction = logits.topk(self.input_size)
        
        return prediction[0, 0].tolist(), (state_h, state_c)
        
        
