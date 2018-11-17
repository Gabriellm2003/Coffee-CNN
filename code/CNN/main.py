import sys
import datamanager as manager
from os import listdir
import tensorflow as tf
import pixelwise as pixelwise
import threshold as thresh
import numpy as np


def print_params(list_params):
    print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for i in range(1, len(sys.argv)):
        print (list_params[i - 1] + '= ' + sys.argv[i])
    print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')





#python main.py Data/ 1 Output/ 25 129 0.005 0.01 1000 100 Train False 
#python main.py Data/ 1 Output/ 25 129 0.005 0.01 1000 100 Test False

def main ():

	list_params = ['data_path', 'fold', 'output_path' ,'window_size', 'batch_size', 'weight_decay', 'initial_learning_rate', 'number_iterations', 'display_step', 'mode[Train/Test]', 'open_set[True/False]']
	if len(sys.argv) < len(list_params) + 1:
		sys.exit('Usage: ' + sys.argv[0] + ' ' + ' '.join(list_params))
	print_params(list_params)

	index = 1
	data_path = sys.argv[index]
	index += 1
	fold = int(sys.argv[index])
	index += 1
	output_path = sys.argv[index]
	index += 1
	window_size = int(sys.argv[index])
	index += 1
	batch_size = int(sys.argv[index])
	index += 1
	weight_decay = float(sys.argv[index])
	index += 1
	initial_learning_rate = float(sys.argv[index])
	index += 1
	number_iterations = int(sys.argv[index])
	index += 1
	display_step = int(sys.argv[index])
	index += 1
	mode = sys.argv[index]
	index += 1
	open_set = sys.argv[index]

	output_path = output_path + str(fold) + "/"

	if (open_set):
		num_classes = 1
	else:
		num_classes = 2
	
	# GPU max memory usage config
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.47)







	if (mode == 'Train'):
		print ("......LOADING IMAGES......")
		images_list, masks_list = manager.makeImageList(data_path, fold, True)
		images = manager.loadImages(images_list, False, window_size)
		masks = manager.loadImages(masks_list, True, window_size)
		print ("......DONE......")

		print ("......PREPARING DATA......")
		mean =  np.array([0.6322493, 0.35163549, 0.33883786])
		std = np.array([0.6322493, 0.35163549, 0.33883786])
		manager.normalizeImages(images, mean, std)
		#total_images = len (images_list)
		if (open_set == 'True'):
			if (('class' + str(fold) + '.txt') not in listdir(output_path)):
				class_pixels_list = manager.makeClassList(masks, output_path, 500, 500, fold)
			else:
				class_pixels_list = manager.makeClassListFromFile (output_path, fold)
		print ("......DONE......")
		#Defining parameters
	
		n_input = window_size * window_size * 3 #RGB
		dropout = 0.5
		#Defining input graph
		x = tf.placeholder (tf.float32, [None, n_input]) 
		y = tf.placeholder(tf.int32)
		batch_prob = tf.placeholder (tf.float32)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		#Defining convolutional network
		logits = pixelwise.pixelwise25(x, batch_prob, True, weight_decay, window_size, 3, open_set)
		loss = pixelwise.create_lossFunc (logits, y) #definindo a funcao loss com cross entropy
		lr = tf.train.exponential_decay(initial_learning_rate, global_step, 50000, 0.1, staircase=True) 
		optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss, global_step=global_step) #otimizador
		#Avaliando o modelo
		correct = tf.nn.in_top_k(logits, y, 1)
		acc_mean = tf.reduce_sum(tf.cast(correct, tf.int32))
		#pred = tf.argmax(logits, dimension = 1)	
		#Inicializando as variaveis
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		restore_saver = tf.train.Saver()




		
		total_images = len(images_list)
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			print ("......STARTING OPTIMIZATION......")
			if str(fold) +'model_final.meta' in listdir(output_path):
				print ("......Model Located: Loading......")
				restore_saver.restore(sess, output_path + str(fold) + 'model_final')
				print ("......Model Restored!!!......")
			else:
				print ("......There isn't a model......")
				sess.run(init)		
			for step in range(1,number_iterations+1):
				if (open_set == 'True'):
					batch_inputs, batch_labels = manager.createTrainBatchPatches_openSet(class_pixels_list, images, total_images, masks,batch_size, window_size)
				else:
					batch_inputs, batch_labels = manager.createTrainBatchPatches(images, masks, total_images, 524, 524, batch_size, window_size)
				batch_inputs = np.reshape(batch_inputs,(-1, n_input))
				batch_loss, output_layer = sess.run([loss,logits], feed_dict={x: batch_inputs, y: batch_labels, batch_prob: dropout})

				if step != 0 and step % display_step == 0:
					predictions = thresh.calculatePredictions(output_layer, 0.5, open_set, False)
					corrects, confusion_matrix = thresh.calculateStatistics(predictions,batch_labels, open_set)
					#print ("SOFTMAX: " + str(output_layer))
					#print ("BATCH CORRECT: " + str(batch_labels))
					#print ("BATCH PREDICTED: " + str(predictions))
					print("Iter " + str(step) + " -- Training Minibatch: Loss= " + "{:.6f}".format(batch_loss) +
						" Accuracy= " + "{:.4f}".format(corrects/batch_size) +
						" Confusion Matrix= " + str(confusion_matrix)
						)
					
			print("Optimization Finished!")

			saver.save(sess, output_path + str(fold) + 'model_final')
			








	if (mode == 'Test'):
		print ("......LOADING IMAGES......")
		test_images_list, test_masks_list = manager.makeImageList(data_path, fold, False)
		test_images = manager.loadImages(test_images_list, False, window_size)
		test_masks = manager.loadImages(test_masks_list, True, window_size)
		print ("......DONE......")
		print ("......PREPARING DATA......")
		mean =  np.array([0.6322493, 0.35163549, 0.33883786])
		std = np.array([0.6322493, 0.35163549, 0.33883786])
		manager.normalizeImages(test_images, mean, std)


		print ("......DONE......")

		#Defining parameters
		n_input = window_size * window_size * 3 #RGB
		dropout = 0.5
		#Defining input graph
		x = tf.placeholder (tf.float32, [None, n_input]) 
		y = tf.placeholder(tf.int32)
		batch_prob = tf.placeholder (tf.float32)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		#Defining convolutional network
		logits = pixelwise.pixelwise25(x, batch_prob, False, weight_decay, window_size, 3, open_set)
		loss = pixelwise.create_lossFunc (logits, y) #definindo a funcao loss com cross entropy
		lr = tf.train.exponential_decay(initial_learning_rate, global_step, 50000, 0.1, staircase=True) 
		optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss, global_step=global_step) #otimizador
		#Avaliando o modelo
		correct = tf.nn.in_top_k(logits, y, 1)
		acc_mean = tf.reduce_sum(tf.cast(correct, tf.int32))
		#pred = tf.argmax(logits, dimension = 1)	
		#Inicializando as variaveis
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		restore_saver = tf.train.Saver()
		total_images = len (test_images_list)

		
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			print ("......Restoring model......")
			restore_saver.restore(sess, output_path + str(fold) + 'model_final') 
			print ("......Model Restored......")
			
			print ("......Generating Images......")
			for i in range(0, len(test_images)):
				print ("......Image "+ str(i) + "......")
				print ("Image: " + str(test_images_list[i]))
				print (".......GENERATING PATCHES.......")
				patches, classes, pos = manager.createPatchesForTest(test_images[i], test_masks[i], window_size)
				print (".......DONE.......")
				thresh.thresholdEstimation_BruteForce (sess, patches, classes, n_input, batch_size, x, y, batch_prob, output_path, logits, i, open_set, num_classes, posPixel=pos)
				print ("......Image " + str(i) + " generated......")
			print ("......Done......")
		
if __name__ == "__main__":
	main()

