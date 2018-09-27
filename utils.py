import numpy as np
import random
import os
from PIL import Image
from matplotlib import pyplot as plt

plt.switch_backend('agg')

config = {}
# number of training samples
n_samples = 0
# generate training data
index = 0
training_sample_list = []

def load_dataset(path, cfg, classifier=True):
	global n_samples
	global config
	global training_sample_list
	global index
	index = 0
	dataset = {}
	config = cfg
	n_samples = config['n_samples']
	f = open(path + '/points.txt', 'r')
	dataset['label'] = eval(f.read())
	dataset['classification'] = []
	dataset['path'] = path
	dataset['classifier'] = classifier
	f.close()
	if classifier:
		f = open(path + '/labels.txt', 'r')
		raw = f.read()
		f.close()
		for i in raw:
			dataset['classification'].append(int(i) - 1)
	training_sample_list = [i for i in range(0, n_samples)]
	random.shuffle(training_sample_list)
	return dataset

def gen_omniglot(batch_size, path):
	def pad0(vector, pad_width, iaxis, kwargs):
			vector[:pad_width[0]] = 1.0
			vector[-pad_width[1]:] = 1.0
			return vector
	images = []
	paths = []
	for i in range(batch_size):
		img_path = path
		while img_path[-3:len(img_path)] != 'png':
			dirs = []
			for i in os.listdir(img_path):
				dirs.append(i)
			a, = random.sample(dirs, 1)
			img_path = os.path.join(img_path, a)
		paths.append(img_path)
		image = Image.open(img_path)
		image = image.resize((128, 128), Image.ANTIALIAS)
		image = np.asarray(image)
		image = np.lib.pad(image, 64, pad0)
		images.append(0.4*(1.0 - image))
	return np.expand_dims(np.array(images), axis=-1), paths

def shuffle_dataset():
	random.shuffle(training_sample_list)

def array2Image(array):
	return Image.fromarray(np.squeeze(array * 255)).convert('RGB')

def generate_batch(dataset, batch_size=None, channels=1, classification=False, inverse=True):
	global training_sample_list
	global index
	global n_samples
	if batch_size == None:
		batch_size = config['batch_size']
	subscript = [i for i in range(index, index + batch_size)]
	if index + batch_size >= n_samples:
		subscript[n_samples - index : batch_size] = [i for i in range(0, index + batch_size - n_samples)]
		index = 0
		random.shuffle(training_sample_list)
	index += batch_size
	batch = [training_sample_list[i] for i in subscript]
	images = []
	class_labels = []
	label_data = []
	label_points = []
	for i in batch:
		img = Image.open('./%s/%d.png' % (dataset['path'], i))
		images.append(np.array(img)[:, :, 0:channels])
		label = dataset['label'][i]
		color = [label[0]['color']]*3 if type(label[0]['color']) is float else label[0]['color'] #[0 if label[0]['color'] < 0 else label[0]['color']] * 3
		label_data.append([np.array([label[0]['radius']] + color)])
		a = np.array([[t['x'], t['y'], t['pressure'], 1] for t in label[1:len(label)+1]])
		if classification:
			onehot = [0 for j in range(0, config['n_classes'])]
			class_label = dataset['classification'][i]
			onehot[class_label] = 1
			class_labels.append(onehot)
		if a.shape[0] < config['max_pts']:
			stop = np.zeros((config['max_pts']-a.shape[0], a.shape[1]), dtype=np.float32)
			stop[:, 2] =  0
			a = np.vstack((a, stop))
		label_points.append(a)

	imgs = np.array(images)
	if inverse:
		imgs = np.ones(imgs.shape) - imgs / 255.0
	else:
		imgs = imgs / 255.0

	if classification:
		return imgs, np.array(class_labels)
	else:
		data = np.squeeze(np.array(label_data))
		points = np.array(label_points)
		data = np.expand_dims(data, 1)
		labels = np.concatenate([data, points], axis=1)
		return imgs, labels

def points_plot(stroke, name, length, n=1):
	fig, ax = plt.subplots()
	plt.axis([-1,1,-1,1])
	color = ['maroon', 'red', 'salmon', 'coral', 'chocolate', 'bisque', \
			 'floralwhite', 'gold', 'olive', 'yellow', 'yellowgreen', 'seagreen', \
			 'turquoise', 'deepskyblue', 'navy', 'slateblue']
	for i in range(0, length * n):
		ax.scatter(stroke[i][0], stroke[i][1], color = color[i//n])
	plt.savefig(name)
	plt.close()
