import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import requests
import re
import time
import base64
import random
import subprocess
from io import BytesIO
from PIL import Image
from Environment import Environment

class Mass(object):
	def __init__(self, mass):
		self.v = np.array([0.0, 0.0, 0.0])
		self.pos = np.array([0.0, 0.0, 0.0])
		self.mass = mass
		self.trajectory = []

	def setVelocity(self, v):
		self.v = np.array(v)

	def setPos(self, p):
		self.pos = np.array(p)

class World(object):
	def __init__(self, objects, G):
		self.objects = objects
		self.G = G

	def update(self, dt):
		G = self.G
		center = np.array([0.0, 0.0, 0.0])
		for i in self.objects:
			force = np.array([0.0, 0.0, 0.0])
			i.trajectory.append(np.array(i.pos))
			for j in self.objects:
				vec = j.pos - i.pos
				r2 = np.sum(np.square(vec))
				f = G*i.mass*j.mass / r2 if r2 > 3e-2 else G*i.mass*j.mass / 3e-2
				f = f * vec / np.linalg.norm(vec) if r2 > 1e-7 else np.array([0.0, 0.0, 0.0])
				force += f
			a = force / i.mass
			i.v += a * dt
			i.pos += i.v * dt
		# keep in center
		for i in self.objects:
			center += i.pos
		center /= len(self.objects)
		for i in self.objects:
			i.pos -= center

	def outBound(self):
		def posbound(v):
			return v < -1.1 or v > 1.1

		for i in self.objects:
			if posbound(i.pos[0]) or posbound(i.pos[1]) or posbound(i.pos[2]):
				return True

	def tooLong(self):
		for i in self.objects:
			if len(i.trajectory) > 2000:
				return True
		return False

world = World([Mass(1.0) for i in range(0, 3)], 0.00005)
world.objects[0].setPos([-0.4, 0.0, 0.2])
world.objects[1].setPos([0.5, 0.2, 0.21])
world.objects[2].setPos([-0.1, 0.6, 0.29])

fig, ax = plt.subplots()
plt.axis([-1, 1, -1, 1])

environment = Environment()
environment.setUrl('http://localhost:3000')
file_cnt = 0
dataset = []

def generate():
	global file_cnt
	global dataset
	if world.outBound() or world.tooLong():
		for i in range(0, 3):
			if len(world.objects[i].trajectory) < 32:
				continue
			else:
				step = 8
				for idx in range(16, len(world.objects[i].trajectory) - 16 * step, 32):
					environment.setSize(256, 256)
					t = random.random()
					t = 0.1 if t < 0.1 else t
					color = [t] * 3
					radius = random.random() / 8
					if radius < 0.002:
						radius = 0.002
					points = []
					for j in range(0, 16):
						# sample from trajectory
						p = 1 / (1 + np.exp(5 * world.objects[i].trajectory[idx][2]))
						points.append({
							'x': world.objects[i].trajectory[idx][0],
							'y': world.objects[i].trajectory[idx][1],
							'pressure': p
						})
						idx += step
					environment.setRadius(radius)
					environment.setColor(color)
					environment.stroke(points)
					environment.getImage().save('./threebody/%d.png' % (file_cnt))
					environment.close()
					data = [{'color': color, 'radius': radius}] + points
					dataset.append(data)
					file_cnt += 1

		for i in range(0, 3):
			world.objects[i].setPos([random.random()*2.2-1.1, random.random()*2.2-1.1, random.random()])
			world.objects[i].setVelocity([0.0, 0.0, 0.0])
			world.objects[i].trajectory = []
	obj = world.objects
	world.update(1)

'''
	scat = []
	for i in range(0, 3):
		scat.append(ax.scatter(obj[i].pos[0], obj[i].pos[1], c = 1, alpha = 1))
	return scat

anim = animation.FuncAnimation(fig, animate, interval=20, blit=True)
plt.show()
'''
print(world)

while file_cnt < 65537:
	generate()
	print(file_cnt)
	if file_cnt % 4096 == 0:
		f = open('./threebody/strokes.txt', 'w')
		f.write(str(dataset))
		f.close()
