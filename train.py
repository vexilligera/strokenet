import torch
import torchvision
from torchvision import transforms
import os
import numpy as np
from utils import *
from model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_coord_encoder(path):
    coordenc = CoordinateEncoder().to(device)
    coorddata = CoordinateData()
    if os.path.exists(path):
        coordenc.load_state_dict(torch.load(path, map_location=device))
    MSE = torch.nn.MSELoss(reduce=False, size_average=False).to(device)
    iter = int(1e6)
    iter_save = 1e4
    lr = 5e-4
    optimizer = torch.optim.Adam(coordenc.parameters(), lr=lr)
    for i in range(iter):
        points, bitmap = coorddata.nextBatch()
        pred = coordenc.forward(points.to(device))
        loss = MSE(bitmap.to(device), pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % iter_save == 0:
            lr *= 0.99
            optimizer = torch.optim.Adam(coordenc.parameters(), lr=lr)
            print('Coordinate encoder loss %f at iteration %d'
                % (loss.cpu().detach().numpy(), i))
            tensor2Image(pred[0, 0], 'bitmap.bmp')
            tensor2Image(bitmap[0, 0], 'bitmap_gt.bmp')
            torch.save(coordenc.state_dict(), path)

def train_generator(coordenc_path, gen_path, dataset_path):
    tb = Threebody(dataset_path, batch_size=32)
    sg = StrokeGenerator(coordenc_path=coordenc_path).to(device)
    sg.coordEncoder.freeze()
    if os.path.exists(gen_path):
        sg.load_state_dict(torch.load(gen_path, map_location=device))
    if not os.path.exists('./gen_output'):
        os.mkdir('./gen_output')
    
    MSE = torch.nn.MSELoss(reduce=False, size_average=False).to(device)
    lr = 5e-5
    optimizer = torch.optim.Adam(filter(
                lambda p: p.requires_grad, sg.parameters()), lr=lr)
    while tb.epoch < 5:
        images, data, points = tb.nextBatch()
        output = sg(torch.Tensor(data).to(device), torch.Tensor(points).to(device))
        loss = MSE(torch.Tensor(images).to(device), output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if tb.iteration % 400 == 0:
            print('\n\rEpoch %d iteration %d loss %f'
                % (tb.epoch, tb.iteration, loss.cpu().detach().numpy()), end='')
            tensor2Image(output[0, 0, :, :], './gen_output/gen_%d%d.bmp' % (tb.epoch, tb.iteration))
            tensor2Image(images[0, 0, :, :], './gen_output/img_%d%d.bmp' % (tb.epoch, tb.iteration))
        if tb.iteration % 2000 == 0:
            torch.save(sg.state_dict(), gen_path)
        if tb.iteration % 5 == 0:
            print('\rEpoch %d iteration %d loss %f'
               % (tb.epoch, tb.iteration, loss.cpu().detach().numpy()), end='')

def train_agent_mnist(gen_path, mnist_agent_path, restrain=True):
    size, channel = 256, 1
    batch_size = 32
    sg = StrokeGenerator().to(device)
    sg.load_state_dict(torch.load(gen_path, map_location=device))
    sg.freeze()
    agent = Agent(size, channel)
    if os.path.exists(mnist_agent_path):
        agent.load_state_dict(torch.load(mnist_agent_path, map_location=device))

    # MNIST preprocess
    mnist_size = 28
    mnist_resize = 120
    brightness = 0.6
    trans = transforms.Compose(transforms = [
        transforms.Resize(mnist_resize),
        transforms.Pad(int((256 - mnist_resize) / 2)),
        transforms.ToTensor(),
        lambda x: x * brightness
    ])
    train_data = torchvision.datasets.MNIST(root='./dataset/mnist',train=True,
                            transform=trans, download=True)
    data_loader = torch.utils.data.DataLoader(train_data, 
                                            batch_size=batch_size, shuffle=True)

    # render program, could also be deployed locally at http://localhost:3000
    renderer = Renderer('http://10.11.6.118:3000', size)
    
    MSE = torch.nn.MSELoss(reduce=False, size_average=False).to(device)
    mse = torch.nn.MSELoss(reduce=False, size_average=False).to(device)
    lr = 1e-4
    LAMBDA = 2e2
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    regularizer = torch.tensor([[1, 1, 1, 0.1]] * batch_size).to(device)
    for epoch in range(10):
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            color_radius, action = agent(images)

            if restrain:
                color_radius = color_radius * regularizer

            approx = sg(color_radius[:, 2:4], action)
            penalty = mse(action[:, 0:15], action[:, 1:16])
            loss = MSE(images, approx) + penalty * LAMBDA
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                tensor2Image(approx[0, 0], './agent_output/%d_approx.bmp' % i)
                tensor2Image(images[0, 0], './agent_output/%d_mnist.bmp' % i)
                data = color_radius[0].cpu().detach().numpy()
                points = action[0].cpu().detach().numpy().tolist()
                data = [1.0 - data[2]] * 3 + [data[3]]
                image = renderer.render(data, points)
                image.save('./agent_output/%d_render.png' % i)
                f = open('./agent_output/%d_data.txt' % i, 'w')
                f.write(str([data] + points))
                f.close()
                print('Iteration %d loss %f' % (i, loss.cpu().detach().numpy()))
        print('Epoch ', epoch)
        torch.save(agent.state_dict(), mnist_agent_path)

def train_recurrent_agent_mnist(gen_path, agent_path):
    batch_size = 32
    rnn_steps = 3
    # MNIST preprocess
    mnist_size = 28
    mnist_resize = 120
    brightness = 0.6
    trans = transforms.Compose(transforms = [
        transforms.Resize(mnist_resize),
        transforms.Pad(int((256 - mnist_resize) / 2)),
        transforms.ToTensor(),
        lambda x: x * brightness
    ])
    train_data = torchvision.datasets.MNIST(root='./dataset/mnist',train=True,
                            transform=trans, download=True)
    data_loader = torch.utils.data.DataLoader(train_data, 
                                            batch_size=batch_size, shuffle=True)

    # model definition and hyper-parameters
    regularizer = torch.tensor([1, 1, 1, 0.1]).to(device)
    # recurrent generator records and composite the canvas at each step
    rg = RecurrentGenerator(1, gen_path).to(device)
    # recurrent agent reads data from rg at each step
    ra = RecurrentAgent(256, 1, rg, max_steps=rnn_steps).to(device)

    if os.path.exists(agent_path):
        ra.load_state_dict(torch.load(agent_path, map_location=device))
    
    MSE = torch.nn.MSELoss(reduce=False, size_average=False).to(device)
    lr = 1e-4
    LAMBDA = 2e2
    optimizer = torch.optim.Adam(filter(
                lambda p: p.requires_grad, ra.parameters()), lr=lr)

    for epoch in range(10):
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            approx, penalty = ra(images, regularizer=regularizer)
            loss = MSE(images, approx) + LAMBDA * penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 200 == 0:
                tensor2Image(approx[0, 0], './agent_output/%d_approx.bmp' % i)
                tensor2Image(images[0, 0], './agent_output/%d_mnist.bmp' % i)
                saveData(ra.steps, rnn_steps, './agent_output/%d_data.txt' % i)
            print('\rIteration %d loss %f' % (i, loss.cpu().detach().numpy()), end='')
        torch.save(ra.state_dict(), agent_path)

# train_coord_encoder('./model/coordenc.pkl')
# train_generator('./model/coordenc.pkl', './model/gen.pkl', './dataset/3')
# train_agent_mnist('./model/gen.pkl', './model/mnist_agent.pkl')
train_recurrent_agent_mnist('./model/gen.pkl', './model/recurrent_mnist_agent.pkl')
