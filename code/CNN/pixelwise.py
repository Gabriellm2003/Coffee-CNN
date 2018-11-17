from tensorflow.python.framework import ops
import tensorflow as tf
import datamanager as manager
import threshold as thresh
import numpy as np



def pixelwise25 (x, keep_prob, is_training, weight_decay, window_size, channels, openSet, debug=False):
	if (openSet == 'True'):
		num_classes = 1 #coffee
	else: 
		num_classes = 2  #coffee and non coffee

	# Input Layer
	
	x = tf.reshape(x, shape=[-1, window_size, window_size, channels])  #redimensionando o tensor

	###CONVOLUCAO####
	# Primeira camada de Convolucao
	conv1 = tf.layers.conv2d(x, filters = 64, kernel_size = [4,4], kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32), 
							 kernel_regularizer = tf.contrib.layers.l2_regularizer(weight_decay), activation = tf.nn.relu, use_bias = True , 
							 bias_initializer = tf.constant_initializer(value=0.1))	
	
	#MaxPooling1
	pool1 =  tf.layers.max_pooling2d(inputs = conv1, pool_size = [2,2], strides = 2)
	
	#Segunda camada de Convolucao	
	conv2 = tf.layers.conv2d(pool1, filters = 128, kernel_size = [4,4], kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32), 
							 kernel_regularizer = tf.contrib.layers.l2_regularizer(weight_decay), activation = tf.nn.relu, use_bias = True , 
							 bias_initializer = tf.constant_initializer(value=0.1))
	
	#MaxPooling2
	pool2 =  tf.layers.max_pooling2d(inputs = conv2, pool_size = [2,2], strides = 2)
	
	#Terceira camada de Convolucao	
	conv3 = tf.layers.conv2d(pool2, filters = 256, kernel_size = [3,3], kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32), 
							 kernel_regularizer = tf.contrib.layers.l2_regularizer(weight_decay), activation = tf.nn.relu, use_bias = True , 
							 bias_initializer = tf.constant_initializer(value=0.1))
	
	#MaxPooling3
	pool3 =  tf.layers.max_pooling2d(inputs = conv3, pool_size = [2,2], strides = 2)
		

	###FULLY CONNECTED LAYER 1###
	pool3_flat = tf.reshape(pool3, [-1, 1*1*256])	
	drop1 = tf.nn.dropout(pool3_flat, keep_prob)
	dense1 = tf.layers.dense(inputs=drop1, units=1024, activation=tf.nn.relu, use_bias = True, bias_initializer = tf.constant_initializer(value=0.1), 
							 kernel_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32), 
							 kernel_regularizer = tf.contrib.layers.l2_regularizer(weight_decay), trainable = is_training)

	###FULLY CONNECTED LAYER 2###	
	drop2 = tf.nn.dropout(dense1, keep_prob)
	dense2 = tf.layers.dense(inputs=drop2, units=1024, activation=tf.nn.relu, use_bias = True, bias_initializer = tf.constant_initializer(value=0.1), 
							 kernel_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32), 
							 kernel_regularizer = tf.contrib.layers.l2_regularizer(weight_decay), trainable = is_training)

	###CLASSIFIER LAYER###
	output_layer = tf.layers.dense(inputs=dense2, units=num_classes, activation=tf.nn.relu, use_bias = True, bias_initializer = tf.constant_initializer(value=0.1), 
								   kernel_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32),
								   kernel_regularizer = tf.contrib.layers.l2_regularizer(weight_decay), trainable = is_training)		

	
	prob = tf.nn.softmax(output_layer)
	
	if (debug):
		print ("CONV1 SHAPE: " + str(conv1.shape))
		print ("POOL1 SHAPE: " + str(pool1.shape))		
		print ("CONV2 SHAPE: " + str(conv2.shape))
		print ("POOL2 SHAPE: " + str(pool2.shape))
		print ("CONV3 SHAPE: " + str(conv3.shape))
		print ("POOL3 SHAPE: " + str(pool3.shape))
		print ("OUTPUT LAYER SHAPE: " + str(output_layer.shape))
		print ("SOFTMAX SHAPE: " + str(tf.shape(prob)))


	return prob



def create_lossFunc(logits, labels):

	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	# The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
	return tf.add_n(tf.get_collection('losses'), name='total_loss')


def test(sess, patches, classes, n_input, batchSize, x, y, keep_prob, output_path, logits, threshold, threshold_stats_file, image_index, openSet, num_classes, posPixel=None):
	#print("Testing Iter " + str(step))
	all_predcs = []
	cm_test = np.zeros((num_classes,num_classes), dtype=np.uint32)
	true_count = 0.0
	for i in range(0,int(len(classes)/batchSize)+1):
		bx = np.reshape(patches[i*batchSize:min(i*batchSize+batchSize, len(classes))], (-1, n_input))
		by = classes[i*batchSize:min(i*batchSize+batchSize, len(classes))]
		output_layer = sess.run([logits], feed_dict={x: bx, y: by, keep_prob: 1.})
		
		"""
		for pred1 in output_layer:	
			for pred2, clas in zip(pred1, by):
				print ("OUTPUTLAYER: " + str(pred2) + "----CLASS: " + str(clas))
		"""


		#print (output_layer)
		preds_val = thresh.calculatePredictions(output_layer, threshold, openSet, True)
		all_predcs = np.concatenate((all_predcs,preds_val))

	corrects, confusion_m = thresh.calculateStatistics(all_predcs, classes, openSet)
	#print ("POS PIXEL = " + str(posPixel) )
	manager.createPredictionMap(output_path, all_predcs, posPixel,image_index ,threshold)
	
	_sum = 0.0
	for i in range(len(cm_test)):
		_sum += (cm_test[i][i]/float(np.sum(cm_test[i])) if np.sum(cm_test[i]) != 0 else 0)

	threshold_stats_file.write(str(threshold) + " " + "{:.6f}".format(corrects/float(len(classes))) + " " + str(confusion_m) + "\n")

	print("IMAGE " + str(image_index) +
		" Accuracy= " +"{:.6f}".format(corrects/float(len(classes))) +
						" Confusion Matrix= " + str(confusion_m)
		)

