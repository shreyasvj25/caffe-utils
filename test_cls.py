import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '../caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

test_listfile = '/scratch/16824/data/testlist_class.txt'
result_file = 'cls_results.txt'
cmat = 'confusion_mat.txt'

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net('/home/svjoshi/assignment/task4/test.prototxt',
                '/home/svjoshi/assignment/task4/models/model_iter_30000.caffemodel',
                caffe.TEST)

test_list = np.loadtxt(test_listfile,  str, comments=None, delimiter='\n')
data_counts = len(test_list)
batch_size = net.blobs['data'].data.shape[0]
batch_count = int(np.ceil(data_counts * 1.0 / batch_size))
accuracy = 0

f = open(result_file, 'w')
f1=open(cmat,'w')
print(batch_count)

##
cmat = np.zeros((30,30))

##
for i in range(batch_count):

	out = net.forward()
	print(i)
	for j in range(batch_size):
		id = i * batch_size + j
		if id >= data_counts:
			break

		lbl = int(test_list[id].split(' ')[1])
		fname = test_list[id].split(' ')[0]
		
		prop = out['softmax'][j] 
		pred_lbl = prop.argmax()
		if pred_lbl == lbl:
			accuracy = accuracy + 1
			cmat[lbl][pred_lbl] = cmat[lbl][pred_lbl] + 1
		else:
			cmat[lbl][pred_lbl] = cmat[lbl][pred_lbl] + 1

		f.write(fname)
		f.write('{0: d}'.format(pred_lbl))
		f.write('\n')
		f1.write('{0: d}'.format(lbl))
		f1.write(' ')
		f1.write('{0: d}'.format(pred_lbl))
		f1.write(' ')
		f1.write('\n')
f.close()

##
for i in range(29):
	a=(cmat[i,i]*100.0/np.sum(cmat[i,:]))
	print(a)
	f1.write(format(a))
	f1.write('\n')
##
f1.close()

accuracy = accuracy * 1.0 / ( data_counts) 
print data_counts
print accuracy


