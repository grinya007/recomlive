import torch

class RNN(torch.nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, device = 'cuda'):
        super(__class__, self).__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.device     = torch.device(device)

        self.embed      = torch.nn.Embedding(input_dim, embedding_dim)
        self.rnn        = torch.nn.GRU(embedding_dim, hidden_dim)
        self.do         = torch.nn.Dropout(0.1)
        self.linear     = torch.nn.Linear(hidden_dim, input_dim)
        self.out        = torch.nn.LogSoftmax(dim = 1)

        self.loss       = torch.nn.CrossEntropyLoss()

        self.to(self.device)

        self.optim      = torch.optim.Adagrad(self.parameters(), lr = 0.05)


    def forward(self, x):
        embeds          = self.embed(x)
        rnn_out, _      = self.rnn(embeds.view(len(x), 1, -1))
        do              = self.do(rnn_out)
        raw_pred        = self.linear(do.view(len(x), -1))
        Y               = self.out(raw_pred)

        return Y

    def fit(self, X):
        loss = 0
        for i in range(len(X) - 1):
            x = torch.tensor(X[i], dtype=torch.long, device=self.device).view(1, 1, -1)
            y = torch.tensor(X[i+1], dtype=torch.long, device=self.device).view(-1)

            Y = self.forward(x)
            loss += self.loss(Y, y)

        self.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
        self.optim.step()

        return loss.item()

    def predict(self, x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.long, device=self.device).view(1, 1, -1)
            Y = self.forward(x)
            _, prediction = Y.topk(self.input_dim)
            return prediction[0].tolist()

