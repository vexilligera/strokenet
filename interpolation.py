import torch
import torchvision
from torchvision import transforms
import os
import numpy as np
from utils import *
from model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def interpolate(gen_path, mnist_agent_path, restrain=True):
    size, channel = 256, 1
    batch_size = 2
    sg = StrokeGenerator().to(device)
    sg.load_state_dict(torch.load(gen_path, map_location=device))
    sg.freeze()
    agent = Agent(size, channel)
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
    embedding = []

    # render program, could also be deployed locally at http://localhost:3000
    renderer = Renderer('http://10.11.6.118:3000', size)

    for i, (images, labels) in enumerate(data_loader):
        images = images.to(device)

        color_radius, action = agent(images)
        temp = agent.decoder.embedding[0:1]
        embedding.append(temp)
        tensor2Image(images[0, 0], './agent_output/interpolate/mnist_%d.bmp' % i)
        if i == 1:
            break
            
    regularizer = torch.tensor([[1, 1, 1, 0.1]] * batch_size).to(device)
    for i in range(0, 11):
        k = (i*0.1)*embedding[0] + (1-i*0.1)*embedding[1]
        
        color_radius, action = agent.decoder.decode(k)
        color_radius = color_radius * regularizer

        data = color_radius[0].cpu().detach().numpy()
        points = action[0].cpu().detach().numpy().tolist()
        data = [1.0 - data[2]] * 3 + [data[3]]
        image = renderer.render(data, points)
        image.save('./agent_output/interpolate/%d_render.png' % i)
  
interpolate('./model/gen.pkl', './model/mnist_agent.pkl')
