import numpy as np

class Math():
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def dsigmoid(y):
        return y * (1 - y)
    def tanh(x):
        return np.tanh(x)
    def dtanh(y):
        return 1 - y * y
    def softmax(v):
        return np.exp(v) / np.sum(np.exp(v))

class Parameter():
    def __init__(self, value):
        self.value = value
        self.derivative = np.zeros_like(value)
        self.momentum = np.zeros_like(value)

    def clear_derivative(self):
        self.derivative.fill(0)

    def clip_derivative(self):
        self.derivative = np.clip(self.derivative, -1, 1)

    def update_value(self, lr):
        self.momentum += self.derivative * self.derivative
        self.value -= lr * self.derivative / np.sqrt(self.momentum + 1e-8)

class LSTM():
    def __init__(self, input_size, hidden_size, weight_sd = 0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.z_size = input_size + hidden_size
        
        self.parameters = {}
        self.parameters['Wf'] = Parameter(
            np.random.randn(hidden_size, self.z_size) * weight_sd + 0.5
        )
        self.parameters['bf'] = Parameter(
            np.zeros((hidden_size, 1))
        )
        self.parameters['Wi'] = Parameter(
            np.random.randn(hidden_size, self.z_size) * weight_sd + 0.5
        )
        self.parameters['bi'] = Parameter(
            np.zeros((hidden_size, 1))
        )
        self.parameters['WC'] = Parameter(
            np.random.randn(hidden_size, self.z_size) * weight_sd
        )
        self.parameters['bC'] = Parameter(
            np.zeros((hidden_size, 1))
        )
        self.parameters['Wo'] = Parameter(
            np.random.randn(hidden_size, self.z_size) * weight_sd + 0.5
        )
        self.parameters['bo'] = Parameter(
            np.zeros((hidden_size, 1))
        )
        self.parameters['Wv'] = Parameter(
            np.random.randn(input_size, hidden_size) * weight_sd
        )
        self.parameters['bv'] = Parameter(
            np.zeros((input_size, 1))
        )

    def clear_derivatives(self):
        for p in self.parameters:
            self.parameters[p].clear_derivative()
    def clip_derivatives(self):
        for p in self.parameters:
            self.parameters[p].clip_derivative()
    def update_parameters(self, lr):
        for p in self.parameters:
            self.parameters[p].update_value(lr)

    def predict(self, X, h_prev = None, C_prev = None):
        if isinstance(h_prev, type(None)):
            h_prev = np.zeros((self.hidden_size, 1))
        if isinstance(C_prev, type(None)):
            C_prev = np.zeros((self.hidden_size, 1))

        _, _, _, _, C, _, h, _, Y = self._forward(
            X.reshape(self.input_size, 1), h_prev, C_prev
        )
        
        return Y, h, C
        
    def fit(self, inputs, h_prev = None, C_prev = None, lr = 0.1):
        if isinstance(h_prev, type(None)):
            h_prev = np.zeros((self.hidden_size, 1))
        if isinstance(C_prev, type(None)):
            C_prev = np.zeros((self.hidden_size, 1))

        assert inputs.shape[1] == self.input_size
        
        z_s, f_s, i_s  = [], [], []
        C_bar_s, C_s, o_s = [], [], []
        v_s, y_s, h_s =  [], [], []
        h_s.append(np.copy(h_prev))
        C_s.append(np.copy(C_prev))
        loss = 0
        for t in range(inputs.shape[0] - 1):
            z, f, i, C_bar, C, o, h, v, y = self._forward(
                inputs[t].reshape(self.input_size, 1), h_s[t], C_s[t]
            )
            z_s.append(z)
            f_s.append(f)
            i_s.append(i)
            C_bar_s.append(C_bar)
            C_s.append(C)
            o_s.append(o)
            h_s.append(h)
            v_s.append(v)
            y_s.append(y)
            loss -= np.dot(inputs[t + 1].reshape(1, self.input_size), np.log(y_s[t]))

        dh_next = np.zeros_like(h_s[0])
        dC_next = np.zeros_like(C_s[0])
        for t in reversed(range(inputs.shape[0] - 1)):
            dh_next, dC_next = self._backward(
                inputs[t + 1].reshape(self.input_size, 1), dh_next, dC_next, C_s[t],
                z_s[t], f_s[t], i_s[t], C_bar_s[t],
                C_s[t + 1], o_s[t], h_s[t + 1], v_s[t], y_s[t]
            )

        self.clip_derivatives()
        self.update_parameters(lr)
        self.clear_derivatives()
        
        return loss, h_s[-1], C_s[-1]



    def _forward(self, X, h_prev, C_prev):
        assert h_prev.shape == (self.hidden_size, 1)
        assert C_prev.shape == (self.hidden_size, 1)

        z = np.row_stack((h_prev, X))
        f = Math.sigmoid(np.dot(self.parameters['Wf'].value, z) + self.parameters['bf'].value)
        i = Math.sigmoid(np.dot(self.parameters['Wi'].value, z) + self.parameters['bi'].value)
        C_bar = Math.tanh(np.dot(self.parameters['WC'].value, z) + self.parameters['bC'].value)

        C = f * C_prev + i * C_bar
        o = Math.sigmoid(np.dot(self.parameters['Wo'].value, z) + self.parameters['bo'].value)
        h = o * Math.tanh(C)

        v = np.dot(self.parameters['Wv'].value, h) + self.parameters['bv'].value
        y = Math.softmax(v)

        return z, f, i, C_bar, C, o, h, v, y
    
    def _backward(self, target, dh_next, dC_next, C_prev, z, f, i, C_bar, C, o, h, v, y):
        assert z.shape == (self.z_size, 1)
        assert v.shape == (self.input_size, 1)
        assert y.shape == (self.input_size, 1)

        for param in [dh_next, dC_next, C_prev, f, i, C_bar, C, o, h]:
            assert param.shape == (self.hidden_size, 1)   
        
        dv = np.copy(y)
        dv -= target

        self.parameters['Wv'].derivative += np.dot(dv, h.T)
        self.parameters['bv'].derivative += dv

        dh = np.dot(self.parameters['Wv'].value.T, dv)
        dh += dh_next
        do = dh * Math.tanh(C)
        do = Math.dsigmoid(o) * do
        self.parameters['Wo'].derivative += np.dot(do, z.T)
        self.parameters['bo'].derivative += do

        dC = np.copy(dC_next)
        dC += dh * o * Math.dtanh(Math.tanh(C))
        dC_bar = dC * i
        dC_bar = Math.dtanh(C_bar) * dC_bar
        self.parameters['WC'].derivative += np.dot(dC_bar, z.T)
        self.parameters['bC'].derivative += dC_bar

        di = dC * C_bar
        di = Math.dsigmoid(i) * di
        self.parameters['Wi'].derivative += np.dot(di, z.T)
        self.parameters['bi'].derivative += di

        df = dC * C_prev
        df = Math.dsigmoid(f) * df
        self.parameters['Wf'].derivative += np.dot(df, z.T)
        self.parameters['bf'].derivative += df

        dz = (np.dot(self.parameters['Wf'].value.T, df)
             + np.dot(self.parameters['Wi'].value.T, di)
             + np.dot(self.parameters['WC'].value.T, dC_bar)
             + np.dot(self.parameters['Wo'].value.T, do))
        dh_prev = dz[:self.hidden_size, :]
        dC_prev = f * dC

        return dh_prev, dC_prev

