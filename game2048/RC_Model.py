import torch
import torch.nn as nn
import torch.nn.functional as F

EPOCH = 20
batch_size = 64
TIME_STEP = 4
INPUT_SIZE = 4
LR = 0.001

class RCNN(nn.Module):
    def __init__(self):
        super(RCNN, self).__init__()
        self.rnn = nn.LSTM(input_size=INPUT_SIZE, hidden_size=256, num_layers=4, bidirectional=True, batch_first=True)
        self.rout = nn.Linear(512, 128)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(4, 1), padding=(2, 0))
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(1, 4), padding=(0, 2))
        self.conv3 = nn.Conv2d(1, 1, kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(1, 1, kernel_size=(4, 4), padding=(2, 2))
        self.fc1 = nn.Linear(5 * 5, 24)
        self.fc2 = nn.Linear(24, 128)
        self.batch_norm1 = nn.BatchNorm1d(5 * 5)
        self.batch_norm2 = nn.BatchNorm1d(24)
        self.rcout = nn.Linear(256,4)

        self.initialize()

    def forward(self, x):
        rnn_out, (h_n, h_c) = self.rnn(x, None)
        rnn_out = self.rout(rnn_out[:, -1, :])
        x = x.unsqueeze(dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 5 * 5)
        x = self.batch_norm1(x)
        x = F.relu(self.fc1(x))
        x = self.batch_norm2(x)
        x = F.relu(self.fc2(x))
        out = torch.cat((rnn_out, x), 1)
        out = self.rcout(out)
        return out

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')