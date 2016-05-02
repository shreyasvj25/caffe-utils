import numpy as np
import cv2

##
def vis_square(data,fname):


    """Take an array of shape (n, height, width) or (n, height, width, 3)
           and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = data*255
    #plt.imshow(data); plt.axis('off')
    #plt.imsave()
    cv2.imwrite(fname,data)
    ##

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

#Before Fine tuning#
solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from('/scratch/16824/models/bvlc_reference_caffenet.caffemodel')
filters = solver.net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1),'conv1bft.jpg')

#After Fine tuning#
net1 = caffe.Net('/home/svjoshi/assignment/task4/test.prototxt',
                '/home/svjoshi/assignment/task4/models/model_iter_30000.caffemodel',
                caffe.TEST)

filters = net1.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1),'conv1aft.jpg')