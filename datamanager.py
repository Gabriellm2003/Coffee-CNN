from os import listdir
import numpy as np
import math
import sys
from PIL import Image, ImageOps 
from skimage import img_as_float
from datetime import datetime
import random


def makeImageList(path, fold, isTrain):
	images = []
	masks = []
	if (isTrain):
		path_images = path + str(fold) + "/train/"
	else:
		path_images = path + str(fold) + "/test/"
	for image in listdir(path_images):
		if 'mask' in image:
			masks.append(path_images + image)
		else:
			images.append(path_images + image)
	
	return images, masks

def loadImages(files, isMask, windowSize):
	h,w = Image.open(files[0]).size
	#print("H = " + str(h) + " W = " + str(w))
	if isMask:
		all_data = np.empty([len(files), h,w], dtype=np.float32) #float #np.uint8 #"int32"
	else:
		#all_data = np.empty([len(files), w+cropSize-1,h+cropSize-1,3], dtype=np.float32)
		all_data = np.empty([len(files), h+windowSize-1,w+windowSize-1,3], dtype=np.float32)
	for i in range(len(files)):
		#print i
		try:
			img = Image.open(files[i])
		except IOError:
			print ("Could not open file!")
			print (files[i])

		img.load()
		if not isMask:
			img = manipulateBorder(img, windowSize)
		data = img_as_float(img)
		#print data[0:25,0:25,1]
		#print ("DATA SHAPE = " + str(data.shape))
		if isMask:
			all_data[i,:,:] = np.floor(data+0.5)
		else:
			all_data[i,:,:,:] = data

	return all_data

def manipulateBorder(img, windowSize):
    
    window = int(windowSize/2)

    w,h = img.size
    crop_left = img.crop((0, 0, windowSize, h))
    crop_right = img.crop((w-windowSize, 0, w, h))
    crop_top = img.crop((0, 0, w, windowSize))
    crop_bottom = img.crop((0, w-windowSize, w, h))

    mirror_left = ImageOps.mirror(crop_left)
    mirror_right = ImageOps.mirror(crop_right)
    flipped_top = ImageOps.flip(crop_top)
    flipped_bottom = ImageOps.flip(crop_bottom)
 
    img_border = ImageOps.expand(img, border=window)
    w_new,h_new = img_border.size
    img_border.paste(mirror_left.crop((window+1, 0, windowSize, h)), (0, window, window, h_new-window))
    img_border.paste(mirror_right.crop((0, 0, window, h)), (w_new-window, window, w_new, h_new-window))
    img_border.paste(flipped_top.crop((0, window+1, w, windowSize)), (window, 0, w_new-window, window))
    img_border.paste(flipped_bottom.crop((0, 0, w, window)), (window, w_new-window, w_new-window, h_new))
    
    img_border.paste(flipped_top.crop((0, window+1, window, windowSize)), (0, 0, window, window))
    img_border.paste(flipped_top.crop((w-window, window+1, w, windowSize)), (w_new-window, 0, w_new, window))
    img_border.paste(flipped_bottom.crop((0, 0, window, window)), (0, h_new-window, window, h_new))
    img_border.paste(flipped_bottom.crop((w-window, 0, w, window)), (w_new-window, h_new-window, w_new, h_new))
    
    return img_border

def makeClassList (masks, file_path, imageXdim, imageYdim, fold):
	counter = 0
	class_list = []
	class_file = file (file_path + 'class' + str(fold) + '.txt', 'w')
	for mask in masks:
		for i in range(0,imageXdim):
			for j in range (0, imageYdim):
				if (mask[i][j] == 1.0):
					index = (counter, i, j)
					class_file.write(str(counter) + " " + str(i) + " " + str(j) + "\n")
					class_list.append(index)
		counter += 1
	class_file.close()
	return class_list

def makeClassListFromFile (path, fold):
	filename = path + "class" + str(fold) + '.txt'
	class_file = file(filename, 'r')
	class_list = []

	for line in class_file:
		aux_list = line.split(' ')
		class_list.append((aux_list[0], aux_list[1], aux_list[2].split('\n')[0]))

	return class_list

def computeImageMean(data):
	mean_full = np.mean(np.mean(np.mean(data, axis=0), axis=0), axis=0)
	std_full = np.std(data, axis=0, ddof=1)[0,0,:]

	return mean_full, std_full

def normalizeImages(data, mean_full, std_full):
	data[:,:,:,0] = np.subtract(data[:,:,:,0], mean_full[0])
	data[:,:,:,1] = np.subtract(data[:,:,:,1], mean_full[1])
	data[:,:,:,2] = np.subtract(data[:,:,:,2], mean_full[2])
	
	data[:,:,:,0] = np.divide(data[:,:,:,0], std_full[0])
	data[:,:,:,1] = np.divide(data[:,:,:,1], std_full[1])
	data[:,:,:,2] = np.divide(data[:,:,:,2], std_full[2])

def createTrainBatchPatches_openSet(class_list, images, number_images, masks, batch_size, window_size):
	patches = []
	classes = []
	window_aux = int(window_size/2)
	random.seed(datetime.now())
	for i in range(0,batch_size):
		random_number = random.randint(0, len(class_list)-1)

		image = int(class_list[random_number][0])
		pixelx = int(class_list[random_number][1])
		pixely = int(class_list[random_number][2])
		#print ("IMAGE: " + str(image))
		#print ("PIXELX: " + str(pixelx))
		#print ("PIXELY: "+ str(pixely))

		createPatchesOpenset(images[image], masks[image], window_size, pixelx+window_aux, pixely+window_aux, patches, classes, (random_number%3))

	return np.asarray(patches), np.asarray(classes,dtype=np.int32)


def createPatchesOpenset(image_data, mask_data, window_size, pixelx, pixely, patches, classes, sorted):
	window = int(window_size/2)
	patch = image_data[pixelx-window:pixelx+window+1, pixely-window:pixely+window+1, :]
	patch_class = retrieveClass(mask_data[pixelx-window,pixely-window])
	classes.append(patch_class)
	#print ("SORTED: " + str(sorted))
	if (sorted == 0):
		patches.append(patch)
	if (sorted == 1):
		mirrorLR = np.fliplr(patch)
		patches.append(mirrorLR)
	if (sorted == 2):
		mirrorUP = np.flipud(patch)
		patches.append(mirrorUP)

def createTrainBatchPatches(images, masks, number_images, lim_imageX, lim_imageY, batch_size, window_size):
	patches = []
	classes = []
	window_aux = int(window_size/2)
	random.seed(datetime.now())
	for i in range(0,batch_size):
		image = random.randint(0, number_images-1)
		pixelx = random.randint(window_aux, lim_imageX-(window_aux+1))
		pixely = random.randint(window_aux, lim_imageY-(window_aux+1))
		#print ("IMAGE = " + str(image) + " PIXELX = " + str(pixelx) + " PIXELY = " + str(pixely))
		createPatches(images[image], masks[image], window_size, pixelx, pixely, patches, classes, pixelx%3)

	return np.asarray(patches), np.asarray(classes,dtype=np.int32)


def createPatches(image_data, mask_data, window_size, pixelx, pixely, patches, classes, sorted):
	#print ("IMAGE DATA SHAPE = " + str(image_data.shape))
	#print ("MASK DATA SHAPE = " + str(mask_data.shape))
	window = int(window_size/2)
	patch = image_data[pixelx-window:pixelx+window+1, pixely-window:pixely+window+1, :]
	patch_class = retrieveClass(mask_data[pixelx-window,pixely-window])
	classes.append(patch_class)

	if (sorted == 0):
		patches.append(patch)
	if (sorted == 1):
		mirrorLR = np.fliplr(patch)
		patches.append(mirrorLR)
	if (sorted == 2):
		mirrorUP = np.flipud(patch)
		patches.append(mirrorUP)
	
def retrieveClass(val):
	if val == 1.0:
		current_class = 1
	elif val == 0.0:
		current_class = 0
	else:
		print("ERROR: mask value not binary ", val)

	return current_class

def createPredictionMap(path, all_predcs, pos, threshold, index):
	img = Image.new("1", (500,500), "black")
	#print ("LEN ALL PREDCS: " + str(len(all_predcs)))
	#print ("POS: " + str(pos))
	for i in range(len(all_predcs)):
		#print ("POS [i][0] = " + str(pos[i][0]))
		#print ("POS [i][1] = " + str(pos[i][1]))
		img.putpixel((int(pos[i][1]), int(pos[i][0])), int(all_predcs[i]))
	img.save(path + str(index) + '_predMap_' + str(threshold) +'.jpeg')


def createPatchesForTest(data,mask_data, window_size):
	window = int(window_size/2)
	patches = []
	classes = []
	pos = []

	for j in range(window,len(data)-window):
		for k in range(window,len(data[j])-window):
			patch = data[j-window:j+window+1,k-window:k+window+1,:] ##np.swapaxes(data[i,j-window:j+window+1,k-window:k+window+1,:], 1,2)
			if len(patch) != window_size or len(patch[0]) != window_size:
				print ("Error Patch size not equal window Size", len(patch), len(patch[0]))
			patches.append(patch)
			current_class = retrieveClass(mask_data[j-window][k-window])
			classes.append(current_class)
	
			current_pos = np.zeros((2))
			current_pos[0] = j-window
			current_pos[1] = k-window
			pos.append(current_pos)

	return np.asarray(patches), np.asarray(classes, dtype=np.int32), pos

