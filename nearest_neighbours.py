# to find n nearest neighbours for m images
# here n=10 and m=3
# Author: Shreyas Joshi (shreyasvj25@gmail.com)

import numpy as np
import cv2

import caffe

# Make sure that caffe is on the python path:
caffe_root = '../caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

# specify a test list file, result text and images path
test_listfile = '/scratch/16824/data/testlist_class.txt'
result_file = 'results.txt'
fullpath = '/scratch/16824/data/crop_imgs/'

# specify m image ids
imgID = [0, 1, 2]

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net('/home/svjoshi/assignment/task4/test.prototxt',
                '/home/svjoshi/assignment/task4/models/model_iter_30000.caffemodel',
                caffe.TEST)

test_list = np.loadtxt(test_listfile,  str, comments=None, delimiter='\n')
data_counts = len(test_list)
batch_size = net.blobs['data'].data.shape[0]
batch_count = int(np.ceil(data_counts * 1.0 / batch_size))

featsize = None
imgfeatures = None
features = None

for i in range(batch_count):

	out = net.forward()
	print(i)
	for j in range(batch_size):
		id = i * batch_size + j
		if id >= data_counts:
			break

		#dataFeat = out['pool5'][j]
		dataFeat = net.blobs['fc7'].data[j]
		if(len(dataFeat.shape) > 1):
			dataFeat = dataFeat.flatten()

		print len(dataFeat)
		featsize = dataFeat.shape[0]

		if features is None:
			features = np.zeros([data_counts,featsize]).astype(np.float32)
			imgfeatures = np.zeros([len(imgID),featsize]).astype(np.float32)

		if id in imgID:
			imgfeatures[id,:] = dataFeat

		features[id,:] = dataFeat


responses = np.arange(data_counts).astype(np.float32)
knn = cv2.KNearest()
knn.train(features,responses)

# find n nearest neighbours using knn search
n=10
ret, results, neighbours ,dist = knn.find_nearest(imgfeatures, n)

#print neighbours
neighbours = neighbours.astype(int)
for i in range(n):
	fname = test_list[neighbours[0,i]].split(' ')[0]
	img0 = cv2.imread(fullpath+fname)
	cv2.imwrite('fc7/0/'+str(i)+'.jpg',img0)

	fname = test_list[neighbours[1,i]].split(' ')[0]
	img1 = cv2.imread(fullpath+fname)
	cv2.imwrite('fc7/1/'+str(i)+'.jpg',img1)

	fname = test_list[neighbours[2,i]].split(' ')[0]
	img2 = cv2.imread(fullpath+fname)
	cv2.imwrite('fc7/2/'+str(i)+'.jpg',img2)

#print imgfeatures.shape
#print features.shape
#print len(dataFeat1d)
#print data_counts



