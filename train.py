import tensorflow as tf
import numpy as np
import math
import random
import os
from PIL import Image
from matplotlib import pyplot as plt
from model import *
from utils import *

config = {
	'image_dim': [256, 256, 1],	# width, height, channels

	# parameters required for dataset loading
	'max_pts': 16, # maximum number of points in one stroke
	'label_dim': [1 + 16, 4],
	'batch_size': 64,
	'n_samples': 65536, #557055, # number of samples
	'n_classes': 4, # number of classes of strokes
	'n_dists': 5, # number of mdn distributions for each point

	'classifier_lr': 1e-3,
	'classifier_epochs': 1,
	# generator learning parameters
	'encoder_lr': 1e-3,
	'encoder_lr_decay': 0.8,
	'encoder_iterations': 80000,
	'decoder_lr': 1e-5,
	'decoder_lr_decay': 0.98,
	'decoder_epochs': 1,
	'mdn_lr': 1e-4,
	'mdn_epochs': 4
}

#dataset = load_dataset('./dataset', config, False)
#dataset = load_dataset('strokes', config, False)
#dataset = load_dataset('./threebody0', config, False)
#dataset = load_dataset('./threebody1', config, False)
#dataset = load_dataset('./threebody2', config, False)
#dataset = load_dataset('./threebody3', config, False)
#dataset = load_dataset('./threebody4', config, False)
#dataset = load_dataset('./threebody5', config, False)
#dataset = load_dataset('./threebody6', config, False)

def train_generator(ckpt_path='./generator', encoder_iterations=config['encoder_iterations'], decoder_epochs=5):
	def train_encoder(iterations):
		def gen_batch(shape, batch_size=config['batch_size']):
			points = []
			labels = []
			flat = shape[0] * shape[1]
			for i in range(0, batch_size):
				x = int(random.random() * (shape[0] + 2) - 2)
				y = int(random.random() * (shape[1] + 2) - 2)
				z = gaussian(x, y, shape, norm=2*math.pi)
				x = x / (shape[0] / 2) - 1
				y = 1 - y / (shape[1] / 2)
				points.append([x, y])
				labels.append(z)
			return np.array(points), np.array(labels)

		def gaussian(mu1, mu2, shape, sig1=1, sig2=1, rho=0, norm=1):
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

		tf.reset_default_graph()
		generator = StrokeGenerator(config['batch_size'], config['max_pts'])
		input_points = tf.placeholder(tf.float32, [config['batch_size'], 2])
		encoded = generator.pointEncoder(input_points, mode='train')

		output_dim = [encoded.shape[1], encoded.shape[2]]
		input_labels = tf.placeholder(tf.float32, [config['batch_size']] + output_dim)
		loss = tf.reduce_mean(tf.square(encoded - input_labels)) * 1e7
		learning_rate = tf.placeholder(tf.float32)
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		train = optimizer.minimize(loss)
		saver = tf.train.Saver(tf.global_variables())
		lr = config['encoder_lr']
		loss_hist = []
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(ckpt_path)
			sess.run(tf.global_variables_initializer())
			if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
				loadParameters(sess, ckpt_path, generator.encoderPrefix)
				print('encoder restored.')

			for i in range(0, iterations):
				points, labels = gen_batch(generator.encoder_feature_shape)
				feed_dict = {
					input_points: points,
					input_labels: labels,
					learning_rate: lr
				}
				_, batch_loss, output = sess.run([train, loss, encoded], feed_dict=feed_dict)
				loss_hist.append(batch_loss)

				if i % 2048 == 0:
					print('Loss at iteration %d/%d is %f' % (i, iterations, batch_loss))
					save_path = saver.save(sess, ckpt_path + '/model.ckpt')
				if (i+1) % 10000 == 0:
					lr *= config['encoder_lr_decay']
		f = open('encoder_loss.txt', 'w')
		f.write(str(loss_hist))
		f.close()

	def train_decoder(epochs):
		tf.reset_default_graph()
		generator = StrokeGenerator(config['batch_size'], config['max_pts'])

		label = tf.placeholder(tf.float32, shape=[config['batch_size'], config['max_pts']+1, 4])
		image = tf.placeholder(tf.float32, shape=[config['batch_size']]+config['image_dim'])
		lr = tf.placeholder(tf.float32, name='learning_rate')

		gen = generator.generate(label)
		loss = tf.reduce_mean(tf.nn.l2_loss(image - gen))
		loss +=  tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()]) * 1e-5
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		trainable_vars = [var for var in tf.trainable_variables() if var.name.startswith(generator.dataPrefix)]
		trainable_vars += [var for var in tf.trainable_variables() if var.name.startswith(generator.genPrefix)]
		# after 3 epochs
		trainable_vars += [var for var in tf.trainable_variables() if var.name.startswith(generator.encoderPrefix)]

		with tf.control_dependencies(update_ops):
			optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=trainable_vars)

		learning_rate = config['decoder_lr']
		saver = tf.train.Saver(tf.global_variables())
		os.system('mkdir %s' % 'generator_output')
		loss_hist = []
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(ckpt_path)
			sess.run(tf.global_variables_initializer())
			loadParameters(sess, ckpt_path, generator.encoderPrefix)
			print('encoder loaded.')
			loadParameters(sess, ckpt_path, generator.genPrefix)
			loadParameters(sess, ckpt_path, generator.dataPrefix)
			print('decoder loaded.')

			iteration = config['n_samples'] // config['batch_size']
			for epoch in range(0, epochs):
				shuffle_dataset()
				for i in range(0, iteration):
					images, labels = generate_batch(dataset)
					train_feed_dict = {
						image: images,
						label: labels,
						lr: learning_rate
					}
					batch_loss, _ = sess.run([loss, optimizer], feed_dict=train_feed_dict)
					loss_hist.append(batch_loss)
					if i % 64 == 0:
						print('Iteration %d/%d batch loss %f' % (i, iteration, batch_loss))
						generated = sess.run(gen, feed_dict=train_feed_dict)
						array2Image(generated[0, :, :]).save('./generator_output/output%d%d.bmp' % (epoch, i))
						array2Image(images[0, :, :]).save('./generator_output/label%d%d.bmp' % (epoch, i))
				print('Epoch %d' % epoch)
				f = open('decoder_loss_whole_2.txt', 'w')
				f.write(str(loss_hist))
				f.close()
				saver.save(sess, ckpt_path + '/model.ckpt')
				learning_rate *= config['decoder_lr_decay']

	#train_encoder(config['encoder_iterations'])
	train_decoder(config['decoder_epochs'])

def train_WGAN(epochs, gen_path='./generator', disc_path='./discriminator'):
	tf.reset_default_graph()
	generator = StrokeGenerator(config['batch_size'], config['max_pts'])
	discriminator = StrokeDiscriminator(generator)
	label = tf.placeholder(tf.float32, shape=[config['batch_size'], config['max_pts']+1, 4])
	image = tf.placeholder(tf.float32, shape=[config['batch_size']]+config['image_dim'])
	gen_loss, disc_loss, l2_loss = discriminator.loss(label, image, l2=1e-2)
	generated_image = generator.generate(label)

	gen_var_list = [var for var in tf.trainable_variables() if var.name.startswith(generator.genPrefix)]
	disc_var_list = [var for var in tf.trainable_variables() if var.name.startswith(discriminator.prefix)]

	gen_opt = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0, beta2=0.9).minimize(gen_loss, var_list=gen_var_list)
	disc_opt = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0, beta2=0.9).minimize(disc_loss, var_list=disc_var_list)
	
	saver = tf.train.Saver(tf.global_variables())
	os.system('mkdir %s' % 'generator_output')
	
	with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(disc_path)
			sess.run(tf.global_variables_initializer())
			path = disc_path
			if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
				loadParameters(sess, disc_path, discriminator.prefix)
			else:
				path = gen_path
			loadParameters(sess, path, generator.encoderPrefix)
			loadParameters(sess, path, generator.dataPrefix)
			loadParameters(sess, path, generator.genPrefix)

			iteration = config['n_samples'] // config['batch_size']
			for epoch in range(0, epochs):
				shuffle_dataset()
				for i in range(0, iteration):
					images, labels = generate_batch(dataset)
					train_feed_dict = {
						image: images,
						label: labels,
					}
					batch_gen_loss, batch_disc_loss, mse = sess.run([gen_loss, disc_loss, l2_loss],
																	feed_dict=train_feed_dict)
					if batch_disc_loss < -4.0:
						sess.run(gen_opt, feed_dict=train_feed_dict)
						gen_image = sess.run(generated_image, feed_dict=train_feed_dict)
						array2Image(gen_image[0, :, :]).save('./generator_output/output%d%d.bmp' % (epoch, i))
						array2Image(images[0, :, :]).save('./generator_output/label%d%d.bmp' % (epoch, i))
					sess.run(disc_opt, feed_dict=train_feed_dict)
											
					print('Iteration %d/%d batch_gen_loss %f, batch_disc_loss %f, l2 loss %f' % \
						 (i, iteration, batch_gen_loss, batch_disc_loss, mse))
				print('Epoch %d' % epoch)
				#saver.save(sess, disc_path + '/model.ckpt')

def train_cnnfc(epochs, cnn_path='./CNNEncoder', gen_path='./generator'):
	from tensorflow.examples.tutorials.mnist import input_data
	tf.reset_default_graph()

	input_image = tf.placeholder(tf.float32, [config['batch_size']] + config['image_dim'])

	cnnEncoder = CNNEncoder(config['batch_size'], config['image_dim'])
	generator = StrokeGenerator(config['batch_size'], config['max_pts'])
	color_radius, pos, pres = cnnEncoder.dense(input_image, [4, 2*config['max_pts'], config['max_pts']])

	_color_radius = tf.tile(tf.constant([[0.04, 0.85,  0.85, 0.85]]), [config['batch_size'], 1])
	#_pres = tf.constant([[0.9] * config['max_pts']] * config['batch_size'])
	pos /= 1.5 #2.0

	points = tf.reshape(pos, [-1, config['max_pts'], 2])
	points = tf.concat([points, tf.expand_dims(pres, axis=-1), tf.zeros([config['batch_size'], config['max_pts'], 1])], axis=-1)
	gen_input = tf.concat([tf.expand_dims(_color_radius, axis=1), points], axis=1)

	gen = generator.generate(gen_input)
	l = gen_input.shape[1]
	d = gen_input[:, 1:l-1, 0:3] - gen_input[:, 2:l, 0:3]
	c = 1e5 #1.5e5
	point_loss = c * tf.reduce_mean(tf.square(d))
	loss = tf.nn.l2_loss(gen - input_image) + point_loss

	learning_rate = tf.placeholder(tf.float32)
	trainable_vars = [var for var in tf.trainable_variables() if var.name.startswith(cnnEncoder.prefix)]
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=trainable_vars)

	mnist = input_data.read_data_sets('./mnist')

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		ckpt = tf.train.get_checkpoint_state(cnn_path)
		if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
			loadParameters(sess, cnn_path, cnnEncoder.prefix)
			print('cnn loaded.')
		loadParameters(sess, gen_path, generator.encoderPrefix)
		loadParameters(sess, gen_path, generator.dataPrefix)
		loadParameters(sess, gen_path, generator.genPrefix)

		def gen_batch():
			batch, labels = mnist.train.next_batch(config['batch_size'])
			batch = np.array(batch)
			test_image = np.reshape(batch, [-1, 28, 28])[0:config['batch_size'], :, :]
			image_array = []

			def pad0(vector, pad_width, iaxis, kwargs):
				vector[:pad_width[0]] = 0
				vector[-pad_width[1]:] = 0
				return vector

			for i in range(0, config['batch_size']):
				image = np.lib.pad(test_image[i], 28, pad0) * 0.4
				image_array.append(np.asarray(Image.fromarray(image).resize((256, 256), Image.ANTIALIAS)))
			test_image = np.expand_dims(np.array(image_array), axis=-1)
			return test_image

		cnt = 0
		loss_hist = []
		penalty_hist = []
		#test_image = gen_batch()
		test_image, paths = gen_omniglot(config['batch_size'], './images_background')
		lr = 8e-5
		for epoch in range(0, epochs):
			if epochs % 1 == 0:
				#test_image = gen_batch()
				test_image, paths = gen_omniglot(config['batch_size'], './images_background')
			feed_dict = {
				input_image: test_image,
				learning_rate: lr
			}
			batch_loss, p_loss, output, _ = sess.run([loss, point_loss, gen, optimizer], feed_dict=feed_dict)
			loss_hist.append(batch_loss)
			penalty_hist.append(p_loss)
			if batch_loss < 2800:
				lr = 4e-5
			else:
				lr = 8e-5
			if epoch % 64 == 0:
				array2Image(output[cnt % 64, :, :]).save('./omniglot_output/output%d.bmp' % (epoch))
				array2Image(test_image[cnt % 64, :, :]).save('./omniglot_output/mnist%d.bmp' % (epoch))
				cnt += 1
				print('iteration ', epoch, batch_loss, p_loss)
				input_data = sess.run(gen_input, feed_dict=feed_dict)
			if batch_loss < 1900:
				break
				#print(str(input_data[0].tolist()))
				'''
				enc_output, enc = sess.run([generator.encoderPoints, generator.encoded], feed_dict=feed_dict)
				for i in range(0, 16):
					array2Image(enc_output[4, :, :, i]).save('./enc_output/%d.bmp' % (i))
					array2Image(enc[4, :, :, i]).save('./enc_output/data%d.bmp' % (i))
				'''
		feed_dict = { input_image: test_image }
		input_data, output = sess.run([gen_input, gen], feed_dict=feed_dict)
		for i in range(0, config['batch_size']):
			array2Image(output[i, :, :]).save('./omni_output/output%d.bmp' % (i))
			array2Image(test_image[i, :, :]).save('./omni_output/mnist%d.bmp' % (i))
		f = open('./omni_output/data.txt', 'w')
		f.write(str(input_data.tolist()))
		f.close()
		f = open('./omni_output/loss.txt', 'w')
		f.write(str(loss_hist))
		f.close()
		f = open('./omni_output/penalty.txt', 'w')
		f.write(str(penalty_hist))
		f.close()
		# omniglot
		f = open('./omni_output/paths.txt', 'w')
		f.write(str(paths))
		f.close()

def main(train_gen=False):  
	if train_gen:
		print('Start training StrokeGenerator.')
		train_generator()
		print('Done training Generator.')
	train_cnnfc(8192*4)
  
main()
