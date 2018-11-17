import pixelwise as pixelw


def calculatePredictions (logitPredcs, threshold, openset, isTest):
	predcs = []
	if openset == 'True':

		if isTest == True:
			
			for pred in logitPredcs:
				for prediction in pred:
					if prediction >= threshold:
						predcs.append(1)
					else:
						predcs.append(0)
		else:

			for prediction in logitPredcs:
				if prediction >= threshold:
					predcs.append(1)
				else:
					predcs.append(0)
	if openset == 'False':

		if isTest == True:

			for pred in logitPredcs:
				for prediction in pred:
					if prediction[1] >= threshold:
						predcs.append(1)
					else:
						predcs.append(0)
		if isTest == False:
			#print ("Caso certo")
			for prediction in logitPredcs:
				if prediction[1] >= threshold:
					predcs.append(1)
				else:
					predcs.append(0)

	return predcs
		



def calculateStatistics (predictions, classes, openset):
	
	if (openset == 'True'):
		correct11 = 0
		incorrect10 = 0
		confusion_matrix = []
		for pred, correct in zip(predictions, classes):
			if (pred == correct):
				correct11 = correct11 + 1
			else:
				incorrect10 = incorrect10 + 1

		confusion_matrix.append (correct11)
		confusion_matrix.append (incorrect10)
		
		return correct11, confusion_matrix
	else:
		correct00 = 0
		incorrect01 = 0
		correct11 = 0
		incorrect10 = 0
		confusion_matrix = []
		#print ("LEN (PRED) = " +  str(len(predictions)))
		#print ("LEN (CLASSES) = " +  str(len(classes)))
		for pred, correct in zip(predictions, classes):
			#print ("PRED = " + str(pred))
			#print ("CORRECT = " + str(correct))
			if (pred == correct):
				if (pred == 0):
					correct00 = correct00 + 1
				else:
					correct11 = correct11 + 1
			else:
				if (pred == 0):
					incorrect01 = incorrect01 + 1
				else:
					incorrect10 = incorrect10 + 1

		corrects = correct00 + correct11

		confusion_matrix.append (correct00)
		confusion_matrix.append (incorrect01)
		confusion_matrix.append (correct11)
		confusion_matrix.append (incorrect10)
		return corrects, confusion_matrix




def readThresholdsFile(filename):
	file = open (filename, "r")
	thresholds = []

	for line in file:
		aux = line.split(" ")
		for number in aux:
			thresholds.append(float(number))
		

	return thresholds



#to use this function u need a test function for ur network that uses calculatePredictions () to determine output.
def thresholdEstimation_BruteForce (sess, patches, classes, n_input, batchSize, x, y, keep_prob, outputPath, logits, image_index, openSet, num_classes, posPixel=None):
	
	threshold_stats_file = open (outputPath + "thresholdsStats.txt", "a")

	thresholds = readThresholdsFile("thresholds.txt") #this file contains all thresholds u want to test separated by " ".

	for threshold in thresholds:
		print ("THRESHOLD = " + str(threshold))
		pixelw.test(sess, patches, classes, n_input, batchSize, x, y, keep_prob, outputPath, logits, threshold, threshold_stats_file, image_index, openSet, num_classes, posPixel)
	threshold_stats_file.close()

	thresh_file = open (filepath + "thresholdsStats.txt", 'r')

	accuracy_list = []
	threshold_list = []
	for line in thresh_file:
		split_list = line.split(" ")
		accuracy_list.append(split_list[1])
		threshold_list.append(split_list[0])
	best_accuracy = max(accuracy_list)
	index = accuracy_list.index(best_accuracy)
	best_threshold = threshold_list[index]

	print ("BEST THRESHOLD = " + str(best_threshold))
	print ("BEST ACCURACY = " + str(best_accuracy))
	filename = 'best_threshold.txt'
	best_threshold_file = open (outputPath + filename, 'w')
	best_threshold_file.write(str(best_accuracy)+ " " + str(best_threshold))
	best_threshold_file.close()

