import torch
import torch.nn as nn


class GRU(nn.Module):
    '''
    input shape: (N,4,128)
    '''

    def __init__(self):
        super(GRU, self).__init__()
        self.net = nn.GRU(input_size=4,hidden_size=128,num_layers=2,batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        '''
        :param x: (N,4,128)
        :return:
        '''

        x = x.transpose(1, 2)
        embed, _ = self.net(x)
        out = embed[:, -1, :]
        pred = self.predictor(out)
        return pred


if __name__ == '__main__':
    x = torch.rand(30,4,128)

    net = GRU()
    y = net(x)
    print(x.shape,y.shape)

    num_params = sum(param.numel() for param in net.parameters())
    print(num_params)