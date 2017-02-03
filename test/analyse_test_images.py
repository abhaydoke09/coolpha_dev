caffe_root = '/home/ubuntu/Caffe/caffe-master/'
sys.path.insert(0,caffe_root + 'python')
import caffe

net = caffe.Classifier('../../models/3-layered-caffenet/train_coolphabet_deploy.prototxt', '../../models/3-layered-caffenet/_iter_5000.caffemodel',mean=np.load('../../models/3-layered-caffenet/out.npy').mean(1).mean(1),channel_swap=(2,1,0),raw_scale=255,image_dims=(227, 227))

def get_image_label(img_name):
    if img_name:
        #imgName = request.args['imgName']
        input_image = caffe.io.load_image('/home/ubuntu/coolpha_data/hpl-telugu-iso-char-offline-test/'+img_name)
        #print input_image
        prediction = net.predict([input_image])
        label = prediction[0].argmax()
        return label
    else:
        print "Please provide image name"



test_images = 'hpl-telugu-test-52.txt'
f = open(test_images,'r')
images = test_images.readlines()

for image_name in images:
	print image_name.split(' ')



