import torch as torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CoordinateEncoder(nn.Module):
    def __init__(self):
        super(CoordinateEncoder, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(256, 1024)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(1024, 64*64)
        self.lrelu3 = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.lrelu1(self.fc1(x))
        out = self.lrelu2(self.fc2(out))
        return self.lrelu3(self.fc3(out)).view((-1, 1, 64, 64))

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

class StrokeGenerator(nn.Module):
    def __init__(self, coordenc_path='', max_pts=16):
        super(StrokeGenerator, self).__init__()
        self.max_pts = max_pts
        self.coordEncoder = CoordinateEncoder()
        if coordenc_path != '':
            self.coordEncoder.load_state_dict(
                torch.load(coordenc_path, map_location=device))
        self.data_fc1 = nn.Linear(2, 256)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.data_fc2 = nn.Linear(256, 64*64)
        self.lrelu2 = nn.LeakyReLU(0.2)
        segments = max_pts - 1
        # grouped conv block 1
        self.conv1_1 = nn.Conv2d(segments, segments, 7,
                                padding=3, groups=segments)
        self.conv1_bn1 = nn.BatchNorm2d(segments)
        self.conv1_2 = nn.Conv2d(max_pts, segments, 7, padding=3)
        self.conv1_bn2 = nn.BatchNorm2d(segments)
        self.conv1_lrelu = nn.LeakyReLU(0.2)
        # conv block 2
        self.conv2_1 = nn.Conv2d(segments*2, 256, 5, padding=2)
        self.conv2_lrelu1 = nn.LeakyReLU(0.2)
        self.conv2_2 = nn.Conv2d(256, 256, 5, padding=2)
        self.conv2_lrelu2 = nn.LeakyReLU(0.2)
        self.conv2_bn = nn.BatchNorm2d(256)
        # deconv block 1
        self.deconv1 = nn.ConvTranspose2d(256, 128, 5, stride=2,
                                        padding=2, output_padding=1)
        self.deconv1_lrelu = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv3_lrelu = nn.LeakyReLU(0.2)
        self.conv3_bn = nn.BatchNorm2d(128)
        # deconv block 2
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2,
                                        padding=2, output_padding=1)
        self.deconv2_lrelu = nn.LeakyReLU(0.2)
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)
        self.tanh = nn.Tanh()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, data, points):
        data_feature = self.lrelu1(self.data_fc1(data))
        data_feature = self.lrelu2(self.data_fc2(data_feature))
        data_feature = data_feature.view((-1, 1, 64, 64))
        length = points.shape[1]
        points_flat = points.view((-1, 3))
        coord = points_flat[:, 0:2]
        pressure = points_flat[:, 2].view((-1, 1, 1, 1))
        coord = self.coordEncoder(coord)
        coord *= pressure.expand(-1, 1, 64, 64)
        coord = coord.view((-1, length, 64, 64))
        coord = coord[:, 0:length - 1] + coord[:, 1:length]
        feature = torch.cat((coord, data_feature), 1)
        # grouped conv block 1
        conv1_1 = self.conv1_bn1(self.conv1_1(feature[:, 0:length-1]))
        conv1_2 = self.conv1_bn2(self.conv1_2(feature))
        h = self.conv1_lrelu(torch.cat((conv1_1, conv1_2), 1))
        # conv block 2
        h = self.conv2_lrelu1(self.conv2_1(h))
        h = self.conv2_lrelu2(self.conv2_2(h))
        h = self.conv2_bn(h)
        # upsample blocks * 2
        h = self.deconv1_lrelu(self.deconv1(h))
        h = self.conv3_bn(self.conv3_lrelu(self.conv3(h)))

        h = self.deconv2_lrelu(self.deconv2(h))
        h = self.tanh(self.conv4(h))

        return h

class AgentConvBlock(nn.Module):
    def __init__(self, nin, nout, ksize=3):
        super(AgentConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(nin, nout, ksize, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(nout, nout, ksize, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.pool = nn.AvgPool2d(2)
    
    def forward(self, x):
        h = self.lrelu1(self.conv1(x))
        h = self.lrelu2(self.conv2(h))
        return self.pool(h)

class AgentCNN(nn.Module):
    def __init__(self, channels):
        super(AgentCNN, self).__init__()
        self.down1 = AgentConvBlock(channels, 16)
        self.down2 = AgentConvBlock(16, 32)
        self.down3 = AgentConvBlock(32, 64)
        self.down4 = AgentConvBlock(64, 128)
        self.down5 = AgentConvBlock(128, 256)

    def output_size(self, input_size):
        return int((input_size / 32)**2 * 256)

    def forward(self, x):
        h = self.down2(self.down1(x))
        h = self.down5(self.down4(self.down3(h)))
        return h.view((-1, int(np.prod(h.shape[1:]))))

class AgentFC(nn.Module):
    def __init__(self, nin, n_steps=16):
        super(AgentFC, self).__init__()
        self.n_steps = n_steps
        self.fc1 = nn.Linear(nin, 1024)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(1024, 256)
        self.lrelu2 = nn.LeakyReLU(0.2)
        # brush decoder
        self.fc_color_radius = nn.Linear(256, 4)
        self.sigmoid1 = nn.Sigmoid()
        # coordinate decoder
        self.fc_coord = nn.Linear(256, n_steps*2)
        self.tanh = nn.Tanh()
        # pressure decoder
        self.fc_pressure = nn.Linear(256, n_steps)
        self.sigmoid2 = nn.Sigmoid()

    def decode(self, embedding):
        h = self.lrelu2(self.fc2(embedding))
        color_radius = self.sigmoid1(self.fc_color_radius(h))
        coord = self.tanh(self.fc_coord(h))
        pressure = self.sigmoid2(self.fc_pressure(h))

        action = coord.view((coord.shape[0], self.n_steps, 2))
        pressure = torch.unsqueeze(pressure, 2)
        action = torch.cat((action, pressure), 2)

        return color_radius, action
    
    def forward(self, x):
        h = self.lrelu1(self.fc1(x))
        self.embedding = h
        return self.decode(h)

class Agent(nn.Module):
    def __init__(self, size, channels):
        super(Agent, self).__init__()
        self.cnn = AgentCNN(channels).to(device)
        self.decoder = AgentFC(self.cnn.output_size(size)).to(device)

    def forward(self, x):
        return self.decoder(self.cnn(x))
