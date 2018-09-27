import tensorflow as tf
import numpy as np
import math
import random

def fc(input, w_shape, name='fc'):
	with tf.variable_scope(name) as scope:
		initializer = tf.contrib.layers.xavier_initializer()
		b_shape = [w_shape[-1]]
		w = tf.get_variable('weights', shape=w_shape, initializer=initializer)
		b = tf.get_variable('biases', shape=b_shape, initializer=initializer)
		return tf.matmul(input, w) + b

def deconv2d(input, w_shape, output_shape, strides=[1, 2, 2, 1], name='deconv2d', activation='leaky_relu'):
	with tf.variable_scope(name) as scope:
		initializer = tf.contrib.layers.xavier_initializer()
		w = tf.get_variable('kernel', shape=w_shape, initializer=initializer)
		b = tf.get_variable('biases', shape=[output_shape[-1]], initializer=initializer)
		x = tf.nn.conv2d_transpose(input, w, output_shape, strides=strides) + b
		if activation == 'leaky_relu':
			return tf.nn.leaky_relu(x)
		elif activation == 'relu':
			return tf.nn.relu(x)
		elif activation == 'tanh':
			return tf.nn.tanh(x)
		else:
			return x

def conv2d(input, w_shape, strides=[1, 1, 1, 1], name='conv2d', activation='leaky_relu'):
	with tf.variable_scope(name) as scope:
		initializer = tf.contrib.layers.xavier_initializer()
		w = tf.get_variable('kernel', shape=w_shape, initializer=initializer)
		b = tf.get_variable('biases', shape=[w_shape[-1]], initializer=initializer)
		x = tf.nn.conv2d(input, w, strides, padding='SAME') + b
		if activation == 'leaky_relu':
			return tf.nn.leaky_relu(x)
		elif activation == 'relu':
			return tf.nn.relu(x)
		elif activation == 'sigmoid':
			return tf.nn.sigmoid(x)
		elif activation == 'tanh':
			return tf.nn.tanh(x)
		else:
			return x

def pool_avg(input, name='pool_avg'):
	return tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
							padding='SAME', name=name)

def loadParameters(sess, path, scope):
	var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
	var_list = [var for var in var_list if 'Adam' not in var.name]
	saver = tf.train.Saver(var_list=var_list)
	ckpt = tf.train.get_checkpoint_state(path)
	saver.restore(sess, ckpt.model_checkpoint_path)

class CNNEncoder(object):
	def __init__(self, batch_size, image_dim):
		self.batch_size = batch_size
		self.image_dim = image_dim
		self.prefix = 'CNN_Encoder'

	def encode(self, input):
		channels = self.image_dim[-1]
		with tf.variable_scope(self.prefix) as scope:
			conv1_1 = conv2d(input, [3, 3, channels, 16], name='conv1_1')
			conv1_2 = conv2d(conv1_1, [3, 3, 16, 16], name='conv1_2')
			pool1 = pool_avg(conv1_2, name='pool1')
			conv2_1 = conv2d(pool1, [3, 3, 16, 32], name='conv2_1')
			conv2_2 = conv2d(conv2_1, [3, 3, 32, 32], name='conv2_2')
			pool2 = pool_avg(conv2_2, name='pool2')
			conv3_1 = conv2d(pool2, [3, 3, 32, 64], name='conv3_1')
			conv3_2 = conv2d(conv3_1, [3, 3, 64, 64], name='conv3_2')
			pool3 = pool_avg(conv3_2, name='pool3')
			conv4_1 = conv2d(pool3, [3, 3, 64, 128], name='conv4_1')
			conv4_2 = conv2d(conv4_1, [3, 3, 128, 128], name='conv4_2')
			pool4 = pool_avg(conv4_2, name='pool4')
			shape = int(np.prod(pool4.shape[1:]))
			flat = tf.reshape(pool4, [-1, shape])
			fc6 = tf.nn.tanh(fc(flat, [shape, 1024], name='fc6'))
		return fc6

	def dense(self, input, output_dim):
		fc6 = self.encode(input)
		with tf.variable_scope(self.prefix) as scope:
			fc7 = tf.nn.leaky_relu(fc(fc6, [1024, 256], name='fc7'))
			radius_color = tf.nn.sigmoid(fc(fc7, [256, output_dim[0]], name='fc8'))
			positions = tf.nn.tanh(fc(fc7, [256, output_dim[1]], name='fc9'))
			pressures = tf.nn.sigmoid(fc(fc7, [256, output_dim[2]], name='fc10'))
		return radius_color, positions, pressures

	def classify(self, input, n_classes):
		fc6 = self.encode(input)
		with tf.variable_scope(self.prefix) as scope:
			fc7 = tf.nn.relu(fc(fc6, [1024, 256], name='fc7'))
			logits = tf.nn.leaky_relu(fc(fc7, [256, n_classes], name='pred'))
		return logits

	def classification_loss(self, labels, pred):
		softmax = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=pred)
		loss = tf.reduce_mean(softmax)
		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return loss, accuracy

# auto encoder
class StrokeGenerator(object):
	def __init__(self, batch_size, max_pts):
		self.batch_size = batch_size
		self.max_pts = max_pts
		self.prefix = 'Stroke_Generator'
		self.encoderPrefix = self.prefix + '/point_encoder'
		self.dataPrefix = self.prefix + '/data_encoder'
		self.genPrefix = self.prefix + '/generator'
		self.encoder_feature_shape = (64, 64)

	def pointEncoder(self, input, output_shape=(64, 64), mode='pass'):
		with tf.variable_scope(self.encoderPrefix, reuse=tf.AUTO_REUSE) as scope:
			fc1 = tf.nn.leaky_relu(fc(input, [2, 256], 'fc1'))
			fc2 = tf.nn.leaky_relu(fc(fc1, [256, 1024], 'fc2'))
			if mode == 'train':
				fc3 = tf.nn.leaky_relu(fc(fc2, [1024, 4096], 'fc3'))
			else:
				fc3 = tf.nn.relu(fc(fc2, [1024, 4096], 'fc3'))
			output = tf.reshape(fc3, [-1, 64, 64])
		return output

	def encoder(self, input_z, pressure_sensitive, color_radius):
		# encode input_z[:, 0] with data encoder
		with tf.variable_scope(self.dataPrefix, reuse=tf.AUTO_REUSE) as scope:
			projection = tf.nn.leaky_relu(fc(input_z[:, 0], [int(input_z.shape[-1]), 1024], name='projection'))
			reshape = tf.reshape(projection, [self.batch_size, 32, 32, 1])
			features = deconv2d(reshape, [5, 5, 1, 1], [self.batch_size, 64, 64, 1], name='features')
		# encode input_z[:, 1:] with encoder
		pos = self.pointEncoder(tf.reshape(input_z[:, 1:self.max_pts+1, 0:2], [-1, 2]))
		pos = tf.reshape(pos, [self.batch_size, self.max_pts, 64, 64])
		pos = tf.transpose(pos, perm=[0, 2, 3, 1])

		if pressure_sensitive:
			pressure = tf.expand_dims(tf.expand_dims(input_z[:, 1:, 2], axis=1), axis=1)
			pressure = tf.tile(pressure, [1, 64, 64, 1])
			point = pressure * pos
		else:
			point = pos

		# 2-point guidance concat color radius
		self.encoderPoints = point
		positions = point[:, :, :, 0:self.max_pts-1] + point[:, :, :, 1:self.max_pts]
		if color_radius:
			encoded = tf.concat([positions, features], axis=-1)
		else:
			encoded = positions
		self.encoded = encoded
		return encoded

	def generate(self, input_z, pressure_sensitive=True, color_radius=True):
		self.encoded = self.encoder(input_z, pressure_sensitive, color_radius)
		encoded = self.encoded
		with tf.variable_scope(self.genPrefix, reuse=tf.AUTO_REUSE) as scope:
			# 3 conv layers
			conv1 = conv2d(encoded, [5, 5, int(encoded.shape[-1]), 512], activation='leaky_relu', name='conv1')
			c1bn = tf.layers.batch_normalization(conv1)
			conv2 = conv2d(c1bn, [5, 5, 512, 256], activation='leaky_relu', name='conv2')
			c2bn = tf.layers.batch_normalization(conv2)
			conv3 = conv2d(c2bn, [5, 5, 256, 128], activation='leaky_relu', name='conv3')
			c3bn = tf.layers.batch_normalization(conv3)
			# 2 deconv layers
			deconv1 = deconv2d(c3bn, [5, 5, 64, 128], [self.batch_size, 128, 128, 64], name='deconv1')
			dc1bn = tf.layers.batch_normalization(deconv1)
			deconv2 = deconv2d(dc1bn, [5, 5, 1, 64], [self.batch_size, 256, 256, 1], name='deconv2', activation='tanh')
		self.output = deconv2
		return deconv2

# Conditional WGAN-GP
class StrokeDiscriminator(object):
	def __init__(self, generator):
		self.generator = generator
		self.prefix = 'Stroke_Discriminator'
		self.max_pts = generator.max_pts

	def critics(self, latent, image):
		with tf.variable_scope(self.prefix, reuse=tf.AUTO_REUSE) as scope:
			conv1 = conv2d(image, [5, 5, 1, 64], strides=[1, 2, 2, 1], activation='leaky_relu', name='conv1')
			# latent.shape[-1] = 16
			conv2 = conv2d(conv1, [5, 5, 64, 128-16], strides=[1, 2, 2, 1], activation='leaky_relu', name='conv2')
			features = tf.concat([conv2, latent], axis=-1)
			conv3 = conv2d(features, [5, 5, 128, 256], strides=[1, 2, 2, 1], activation='leaky_relu', name='conv3')
			conv4 = conv2d(conv3, [5, 5, 256, 256], strides=[1, 1, 1, 1], activation='leaky_relu', name='conv4')
			conv5 = conv2d(conv4, [5, 5, 256, 512], strides=[1, 2, 2, 1], activation='leaky_relu', name='conv5')
			shape = int(np.prod(conv5.shape[1:]))
			flat = tf.reshape(conv5, [-1, shape])
			fc6 = tf.nn.leaky_relu(fc(flat, [shape, 1024], name='fc6'))
			fc7 = tf.nn.leaky_relu(fc(fc6, [1024, 1], name='fc7'))
			return fc7

	def loss(self, input_z, gt_image, lmbda=10, l2=0.0):
		generated = self.generator.generate(input_z)
		encode = self.generator.encoded
		true_logits = self.critics(encode, gt_image)
		fake_logits = self.critics(encode, generated)

		mse = tf.reduce_mean(tf.nn.l2_loss(gt_image - generated))
		l2_loss = mse * l2
		gen_loss = -tf.reduce_mean(fake_logits) + l2_loss
		disc_loss = tf.reduce_mean(fake_logits - true_logits)

		alpha = tf.random_uniform([self.generator.batch_size, 1], minval=0, maxval=1)
		shape = int(np.prod(gt_image.shape[1:]))
		alpha = tf.reshape(tf.tile(alpha, [1, shape]), gt_image.shape)
		interpolates = gt_image + (generated - gt_image) * alpha
		grads = tf.gradients(self.critics(encode, interpolates), [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1]))
		gp = tf.reduce_mean(tf.square(slopes - 1.0))
		disc_loss += lmbda * gp 

		return gen_loss, disc_loss, mse

class MDN(object):
	def __init__(self, batch_size, max_pts, n_dist=5):
		self.batch_size = batch_size
		self.max_pts = max_pts
		self.n_dist = n_dist
		self.prefix = 'mdn'

	def gaussian_3d(self, y, params):
		det_recp = tf.reciprocal(params[-1] + 1e-10)
		coeff = (math.pi*2)**(-1.5) * tf.sqrt(det_recp)
		a, b, c, d, e, f = params[0], params[1], params[2], params[3], params[4], params[5]
		mu_1, mu_2, mu_3 = params[6], params[7], params[8]
		exp_coeff = -0.5 * det_recp
		d1 = y[:, 0:1] - mu_1
		d2 = y[:, 1:2] - mu_2
		d3 = y[:, 2:3] - mu_3
		exp_term = a*tf.square(d1) + 2*b*d2*d1 + 2*c*d3*d1 + d*tf.square(d2) + 2*e*d3*d2 + f*tf.square(d3)
		prob = coeff * tf.exp(exp_coeff * exp_term)
		return prob

	def decode(self, input):
		with tf.name_scope(self.prefix) as scope:
			radius_color_fc = tf.nn.leaky_relu(fc(input, [1024, 64], 'radius_color_fc_1'))
			radius_color = tf.nn.leaky_relu(fc(radius_color_fc, [64, 4], 'radius_color_fc_2'))
			outputs = tf.nn.leaky_relu(fc(input, [1024, 1024], 'fc_partition'))
			outputs = tf.reshape(outputs, [-1, self.max_pts, 64])
			
			flat = tf.reshape(outputs, [-1, 64])
			gaussians = fc(flat, [64, 10 * self.n_dist])

			pi = tf.placeholder(dtype=tf.float32, shape=[None,self.n_dist], name="mixparam")
			l1 = tf.placeholder(dtype=tf.float32, shape=[None,self.n_dist], name="mixparam")
			l2 = tf.placeholder(dtype=tf.float32, shape=[None,self.n_dist], name="mixparam")
			l3 = tf.placeholder(dtype=tf.float32, shape=[None,self.n_dist], name="mixparam")
			l4 = tf.placeholder(dtype=tf.float32, shape=[None,self.n_dist], name="mixparam")
			l5 = tf.placeholder(dtype=tf.float32, shape=[None,self.n_dist], name="mixparam")
			l6 = tf.placeholder(dtype=tf.float32, shape=[None,self.n_dist], name="mixparam")
			mu_1 = tf.placeholder(dtype=tf.float32, shape=[None,self.n_dist], name="mixparam")
			mu_2 = tf.placeholder(dtype=tf.float32, shape=[None,self.n_dist], name="mixparam")
			mu_3 = tf.placeholder(dtype=tf.float32, shape=[None,self.n_dist], name="mixparam")

			pi, l1, l2, l3, l4, l5, l6, mu_1, mu_2, mu_3 = tf.split(
				gaussians, num_or_size_splits=10, axis=1)
			# Choleksy decomposition of sigma, l1~l3 diagonal
			l1 = tf.nn.sigmoid(l1) / 10
			l2 = tf.nn.sigmoid(l2) / 10
			l3 = tf.nn.sigmoid(l3) / 10
			l4 = tf.nn.tanh(l4) / 10
			l5 = tf.nn.tanh(l5) / 10
			l6 = tf.nn.tanh(l6) / 10

			cov_xx = l1*l1
			cov_yy = l4*l4 + l2*l2
			cov_zz = l5*l5 +l6*l6 + l3*l3
			cov_xy = l1*l4
			cov_xz = l1*l5
			cov_yz = l4*l5 + l2*l6

			max_pi = tf.reduce_max(pi, 1, keepdims=True)
			pi = pi - max_pi
			pi = tf.exp(pi)
			normalized = tf.reciprocal(tf.reduce_sum(pi, 1, keepdims=True))
			pi = normalized * pi

			mu_1 = tf.nn.tanh(mu_1)
			mu_2 = tf.nn.tanh(mu_2)
			mu_3 = tf.nn.sigmoid(mu_3)

			det_sigma = tf.square(l1 * l2 * l3)
			a = cov_yy*cov_zz - cov_yz*cov_yz
			b = cov_xz*cov_yz - cov_xy*cov_zz
			c = cov_xy*cov_yz - cov_xz*cov_yy
			d = cov_xx*cov_zz - cov_xz*cov_xz
			e = cov_xy*cov_xz - cov_xx*cov_yz
			f = cov_xx*cov_yy - cov_xy*cov_xy
		return [a, b, c, d, e, f, mu_1, mu_2, mu_3, pi, det_sigma], \
			   [cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz],    \
			   radius_color

	def position_loss(self, input_y, params):
		y = tf.reshape(input_y[:, 1:, 0:3], [-1, 3])
		prob = self.gaussian_3d(y, params)
		result = prob * params[-2]
		result = tf.reduce_sum(result, 1, keepdims=True)
		result = -tf.log(tf.clip_by_value(result, 1e-20, 1e10))
		loss = tf.reduce_mean(result)
		return loss

	def sample(self, params, covariance, subscript, n=1, m=0):
		# sample single stroke of #max_pts points from the GMM generated by mdn
		def get_pi_idx(x, pdf):
			accm = 0
			for i in range(0, pdf.size):
				accm += pdf[i]
				if accm >= x:
					return i
			return -1

		pi = params[-2] 
		stroke = []
		l = self.max_pts
		for i in subscript:
			for j in range(0, n):
				idx = get_pi_idx(random.random(), pi[i])
				mu1 = params[6][i+l*m, idx]
				mu2 = params[7][i+l*m, idx]
				mu3 = params[8][i+l*m, idx]
				mean = [mu1, mu2, mu3]
				cov = [[covariance[0][i+l*m, idx], covariance[3][i+l*m, idx], covariance[4][i+l*m, idx]],
					   [covariance[3][i+l*m, idx], covariance[1][i+l*m, idx], covariance[5][i+l*m, idx]],
					   [covariance[4][i+l*m, idx], covariance[5][i+l*m, idx], covariance[2][i+l*m, idx]]]
				x = np.random.multivariate_normal(mean, cov)
				stroke.append(x)
		return stroke

class RNNDecoder(object):
	def __init__(self, batch_size, max_pts):
		self.batch_size = batch_size
		self.max_pts = max_pts

	def rnn(self, input):
		rnn = tf.contrib.rnn
		inputs = tf.expand_dims(input, axis=1)
		inputs = tf.tile(inputs,[1, self.max_pts, 1])
		with tf.name_scope('lstm') as scope:
			fc1 = tf.nn.leaky_relu(fc(input, [1024, 64], 'radius_color_fc_1'))
			self.radius_color = tf.nn.tanh(fc(fc1, [64, 4], 'radius_color_fc2'))
			cells = [rnn.LSTMCell(units) for units in [256, 64]]
			cells = rnn.MultiRNNCell(cells)
			zero_state = cells.zero_state(self.batch_size, dtype=tf.float32)
			rnn_outputs, state = tf.nn.dynamic_rnn(cells, inputs, initial_state=zero_state, dtype=tf.float32)
		return rnn_outputs

	def output(self, rnn_outputs):
		with tf.name_scope('lstm') as scope:
			rnn = tf.reshape(rnn_outputs, [self.batch_size, -1])
			flat = tf.nn.tanh(fc(rnn, [16*64, 3]))
			out = tf.reshape(flat, [self.batch_size, -1])
			rc = self.radius_color
			out = tf.concat([rc, out], axis=1)
		return out
