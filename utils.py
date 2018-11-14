import numpy as np
import math
import random
import pickle
import torch
from PIL import Image
from NodeServer import Draw
from graphviz import Digraph

def tensor2Image(tensor, path='', norm=True):
    t = tensor
    if 'numpy' not in str(type(tensor)):
        t = tensor.cpu().detach().numpy()
    if norm:
        t = t * 255
    if len(tensor.shape) != 3:
        img = Image.fromarray(t).convert('RGB')
    else:
        r = Image.fromarray(t[0]).convert('L')
        g = Image.fromarray(t[1]).convert('L')
        b = Image.fromarray(t[2]).convert('L')
        img = Image.merge("RGB", (r, g, b))
    if path == '':
        img.show()
    else:
        img.save(path)

class Renderer(object):
    def __init__(self, url, size):
        self.url = url
        self.size = size
        self.draw = Draw()
        self.draw.setUrl(url)

    def render(self, color_radius, points):
        draw = self.draw
        stroke = []
        n = len(points)
        for i in range(n):
            x = points[i][0]
            y = points[i][1]
            p = points[i][2]
            stroke.append({'x': x, 'y': y, 'pressure': p})
        draw.setSize(self.size, self.size)
        draw.setRadius(color_radius[-1])
        draw.setColor(color_radius[0:3])
        draw.stroke(stroke)
        image = draw.getImage()
        draw.close()
        return image

class CoordinateData(object):
    def __init__(self, shape=[64, 64], batch_size=64):
        self.shape = shape
        self.batch_size = batch_size

    def gaussian(self, mu1, mu2, sig1=1.0, sig2=1.0, rho=0, norm=1):
        shape = self.shape
        x = np.array([i for i in range(0 - mu1, shape[0] - mu1)])
        y = np.array([i for i in range(0 - mu2, shape[1] - mu2)])
        u = np.tile(x, (shape[1], 1)) / sig1
        v = np.tile(y, (shape[0], 1)).T / sig2
        a = 1 / (2 * math.pi * sig1 * sig2 * np.sqrt(1 - rho**2))
        b = -1 / (2 * (1 - rho**2)) * (np.square(u) - 2 * rho * u * v + np.square(v))
        z = a * np.exp(b)
        if norm != 0:
            z *= norm
        return z

    def nextBatch(self):
        points = []
        bitmap = []
        shape = self.shape
        for i in range(self.batch_size):
            x = int(random.random() * (shape[0] + 2) - 2)
            y = int(random.random() * (shape[1] + 2) - 2)
            z = self.gaussian(x, y, norm=2*math.pi)
            x = x / (shape[0] / 2) - 1
            y = 1 - y / (shape[1] / 2)
            points.append([x, y])
            bitmap.append(z)
        points, bitmap = np.array(points), np.array(bitmap)
        return torch.FloatTensor(points), torch.FloatTensor(bitmap).unsqueeze(1)

class Threebody(object):
    def load(self, path):
        self.path = path
        f = open(path + '/strokes.pkl', 'rb')
        self.data = pickle.load(f)
        self.n_samples = len(self.data)
        f.close()
        self.sample_list = [i for i in range(self.n_samples)]
        random.shuffle(self.sample_list)
        self.index = 0
        self.epoch = 0
        self.iteration = 0

    def __init__(self, path, batch_size=64):
        self.load(path)
        self.batch_size = batch_size
        if batch_size > self.n_samples:
            raise("utils.py: batch size cannot be bigger than #samples.")

    def nextBatch(self):
        images, data, trajectories = [], [], []
        for i in range(self.batch_size):
            if self.index == self.n_samples:
                self.index = 0
                random.shuffle(self.sample_list)
                self.epoch += 1
                self.iteration = 0
            num = self.sample_list[self.index]
            image = Image.open('%s/%d.png' % (self.path, num))
            d = self.data[num]
            images.append(1.0 - np.array(image)[:, :, 0] / 255.0)
            data.append(d[0:2])
            trajectories.append(d[2:])
            self.index += 1
        data = np.array(data)
        trajectories = np.array(trajectories)
        # NCWH
        images = np.expand_dims(np.array(images), 1)
        self.iteration += 1
        return images, data, trajectories

# neural network visualization and gradient debug
# https://gist.github.com/apaszke/f93a377244be9bfcb96d3547b9bc424d
def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen or type(fn) == type(None):
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if type(grad_output) != type(None):
            grad_output = grad_output.data
            return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()
        return True

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot

def saveData(data, n_steps, path, idx=0):
    step_data = []
    for i in range(n_steps):
        color_radius = data[i][0][idx].cpu().detach().numpy().tolist()
        action = data[i][1][idx].cpu().detach().numpy().tolist()
        step_data.append([color_radius] + action)
    f = open(path, 'w')
    f.write(str(step_data))
    f.close()
