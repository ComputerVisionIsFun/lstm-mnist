import torch.nn as nn
import parameters as P
import torch



class model_mnist(nn.Module):
    def __init__(self):
        super(model_mnist, self).__init__()
        
        self.lstm = nn.LSTM(input_size=P.INPUT_SIZE,hidden_size=P.HIDDEN_SIZE, num_layers=P.NUM_LAYERS,batch_first=P.BATCH_FIRST)
        self.linear = nn.Linear(P.L*P.HIDDEN_SIZE, P.N_CLASSES)
        
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.contiguous().view(-1, P.L*P.HIDDEN_SIZE)
        x = self.linear(x)

        return x
