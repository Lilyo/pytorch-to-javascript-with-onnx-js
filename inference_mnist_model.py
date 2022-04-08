import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = 0.1307
STANDARD_DEVIATION = 0.3081


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        x = x.reshape(280, 280, 4)
        x = torch.narrow(x, dim=2, start=3, length=1)
        x = x.reshape(1, 1, 280, 280)
        x = F.avg_pool2d(x, 10, stride=10)


        # # opset_version=9
        # index = torch.tensor([27 for _ in range(28)])
        # index = index.expand(1, 1, 1, 28)
        # self.sliding_win.scatter_add(2, index, x[:, :, :1, :])

        x = x / 255
        x = (x - MEAN) / STANDARD_DEVIATION

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output


##############################
#     Encoder(ResNet-18)
##############################
# https://blog.csdn.net/sunqiande88/article/details/80100891
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, latent_dim=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = self.make_layer(ResidualBlock, 64, 1, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, 64, 1, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 1, stride=2)
        # self.rcan1 = ChannelAttention(64, 16)
        # self.rcan2 = ChannelAttention(64, 16)
        # self.rcan3 = ChannelAttention(64, 16)
        self.fc = nn.Linear(3136, latent_dim)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        # print("conv1: ", out.size())
        out = self.conv2(out)
        # print("conv2: ", out.size())
        out = self.layer1(out)
        # print("layer1: ", out.size())
        out = self.layer2(out)
        # print("layer2: ", out.size())
        out = self.layer3(out)
        # print("layer3: ", out.size())
        out = out.view(out.size(0), -1)
        # print("view: ", out.size())
        out = self.fc(out)
        # print("fc: ", out.size())
        # quit()
        return out

##############################
#           LSTM
##############################

class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim):
        super(LSTM, self).__init__()

        self.layers = num_layers
        self.rnn_drop = 0.1 if self.layers > 1 else 0
        self.hidden_size = hidden_dim

        self.rnn = nn.LSTM(
            input_size=latent_dim,  # The number of expected features in the input x
            hidden_size=self.hidden_size,  # rnn hidden unit
            num_layers=self.layers,  # number of rnn layers
            batch_first=True,  # set batch first
            dropout=self.rnn_drop,  # dropout probability
            bidirectional=False  # bi-LSTM
        )


        # LSTM Initialization,
        for name, params in self.rnn.named_parameters():
            # weight: Orthogonal Initialization
            if 'weight' in name:
                nn.init.orthogonal_(params)
            # lstm forget gate bias init with 1.0
            if 'bias' in name:
                b_i, b_f, b_c, b_o = params.chunk(4, 0)
                nn.init.ones_(b_f)

    def forward(self, x):

        # initialization hidden state
        # 1.zero init
        out, (h_n, c_n) = self.rnn(x, None)  # None represents zero initial hidden state

        return out


##############################
#           MGU
##############################

class MGU(nn.Module):
    def __init__(self, latent_dim=512):
        super(MGU, self).__init__()

        self.fc1_dim = latent_dim

        # MGU parameters
        self.mgu_g = nn.Linear(self.fc1_dim + self.fc1_dim, self.fc1_dim)
        self.mgu_c = nn.Linear(self.fc1_dim + self.fc1_dim, self.fc1_dim)
        self.dp4 = nn.Dropout(1.0 - 1.0)

    # Fixed length MGU by padding zeros
    def forward(self, x):
        mgu_tmp = torch.zeros_like(x)
        for i in range(x.size(1)):
            self.mgu_hidden_state = self.dp4(self.mgu_hidden_state)
            g = torch.cat((x[:, i, :], self.mgu_hidden_state), dim=1)
            g = self.mgu_g(g)
            g = torch.sigmoid(g)

            c = torch.cat((x[:, i, :], g * self.mgu_hidden_state), dim=1)
            c = self.mgu_c(c)
            c = torch.tanh(c)

            self.mgu_hidden_state = (1.0 - g) * self.mgu_hidden_state + g * c
            # self.mgu_hidden_state = g * self.mgu_hidden_state + (1.0 - g) * c

            mgu_tmp[:, i, :] = self.mgu_hidden_state

        return mgu_tmp

    def reset_hidden_state(self, b_size):
        self.mgu_hidden_state = torch.randn(b_size, self.fc1_dim).cuda()


##############################
#         ConvMGU
##############################


class ConvMGU(nn.Module):
    def __init__(self, num_classes, latent_dim=512, lstm_layers=1, hidden_dim=1024):
        super(ConvMGU, self).__init__()
        self.encoder = ResNet(ResidualBlock, latent_dim)

        self.mgu = MGU(hidden_dim)
        self.bn1d1 = nn.BatchNorm1d(latent_dim)
        self.bn1d2 = nn.BatchNorm1d(hidden_dim)
        self.finallayer1 = nn.Linear(hidden_dim, hidden_dim)
        self.finallayer2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        x = x.reshape(112, 112, 4)
        x = torch.narrow(x, dim=2, start=0, length=3)
        x = x.reshape(1, 1, 3, 112, 112)

        batch_size, seq_length, c, h, w = x.shape
        # print(x.shape)
        # quit()
        x = x.reshape(batch_size * seq_length, c, h, w)
        # print(x.shape)
        x = self.encoder(x)
        # print("encoder out", x.shape)
        x = x.reshape(batch_size, seq_length, -1)
        x = self.bn1d1(x.transpose(2, 1)).transpose(2, 1)

        x = self.mgu(x)

        x = self.bn1d2(x.transpose(2, 1)).transpose(2, 1)

        x = x.reshape(batch_size * seq_length, -1)
        # x = F.relu(self.finallayer1(x))
        x = self.finallayer2(x)
        # x = x.reshape(batch_size, seq_length, -1)
        return x



##############################
#           MGU ONNX
##############################

class MGU_ONNX(nn.Module):
    def __init__(self, latent_dim=512):
        super(MGU_ONNX, self).__init__()

        self.fc1_dim = latent_dim

        # MGU parameters
        self.mgu_g = nn.Linear(self.fc1_dim + self.fc1_dim, self.fc1_dim)
        self.mgu_c = nn.Linear(self.fc1_dim + self.fc1_dim, self.fc1_dim)
        self.dp4 = nn.Dropout(1.0 - 1.0)

    # Fixed length MGU by padding zeros
    def forward(self, x, h0):
        self.mgu_hidden_state = h0

        self.mgu_hidden_state = self.dp4(self.mgu_hidden_state)
        g = torch.cat((x, self.mgu_hidden_state), dim=1)
        g = self.mgu_g(g)
        g = torch.sigmoid(g)

        c = torch.cat((x, g * self.mgu_hidden_state), dim=1)
        c = self.mgu_c(c)
        c = torch.tanh(c)

        self.mgu_hidden_state = (1.0 - g) * self.mgu_hidden_state + g * c
        # self.mgu_hidden_state = g * self.mgu_hidden_state + (1.0 - g) * c

        return self.mgu_hidden_state

    def reset_hidden_state(self, b_size):
        self.mgu_hidden_state = torch.randn(b_size, self.fc1_dim).cuda()


class ConvMGU_ONNX(nn.Module):
    def __init__(self, num_classes, latent_dim=512, lstm_layers=1, hidden_dim=1024):
        super(ConvMGU_ONNX, self).__init__()
        self.encoder = ResNet(ResidualBlock, latent_dim)

        self.mgu = MGU_ONNX(hidden_dim)
        self.bn1d1 = nn.BatchNorm1d(latent_dim)
        self.bn1d2 = nn.BatchNorm1d(hidden_dim)
        self.finallayer1 = nn.Linear(hidden_dim, hidden_dim)
        self.finallayer2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, h0):
        # print(h0.shape)
        # quit()
        h0 = h0.reshape(1, 512)
        x = x.reshape(112, 112, 4)
        x = torch.narrow(x, dim=2, start=0, length=3)
        x = x.reshape(1, 1, 3, 112, 112)

        batch_size, seq_length, c, h, w = x.shape
        x = x.reshape(batch_size * seq_length, c, h, w)
        x = self.encoder(x)

        # x = self.bn1d1(x)
        hn = self.mgu(x, h0)
        # x = self.bn1d2(hn)

        x = self.finallayer2(x)

        x = torch.softmax(x, dim=-1)

        return x, hn.reshape(-1)