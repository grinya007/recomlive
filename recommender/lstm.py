import torch
import torch.nn as nn
import torch.optim as optim

class LSTM(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, device = torch.device('cuda')):
        super(__class__, self).__init__()
        
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.device     = device

        self.embed      = nn.Embedding(input_dim, embedding_dim)
        self.lstm       = nn.LSTM(embedding_dim, hidden_dim)
        self.linear     = nn.Linear(hidden_dim, input_dim)
        self.out        = nn.LogSoftmax(dim = 1)
        
        self.loss       = nn.MSELoss()
        self.optim      = optim.SGD(self.parameters(), lr=0.1)
        
        self.to(device)
        

    def _init_hidden(self):
        return (
            torch.zeros(1, 1, self.hidden_dim, dtype=torch.float, device=self.device),
            torch.zeros(1, 1, self.hidden_dim, dtype=torch.float, device=self.device)
        )
    
    def forward(self, x, h = None):
        if isinstance(h, type(None)):
            h = self._init_hidden()
            
        embeds          = self.embed(x)
        lstm_out, h     = self.lstm(embeds.view(len(x), 1, -1), h)
        raw_pred        = self.linear(lstm_out.view(len(x), -1))
        Y               = self.out(raw_pred)
        
        return Y, h
    
    def fit(self, X, h = None):
        loss = 0
        for i in range(len(X) - 1):
            x = torch.tensor(X[i], dtype=torch.long, device=self.device).view(1, 1, -1)
            y = torch.tensor(X[i+1], dtype=torch.long, device=self.device).view(-1)
            
            Y, h = self.forward(x, h)
            loss += self.loss(Y, y)
            
        self.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optim.step()
        h[0].detach()
        h[1].detach()
        
        return loss.item(), h
    
    def predict(self, x, h = None):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.long, device=self.device).view(1, 1, -1)
            Y, h = self.forward(x, h)
            _, prediction = Y.topk(self.input_dim)
            return prediction[0].tolist(), h
