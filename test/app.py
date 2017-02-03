from flask import Flask, url_for
from flask import request
import json 
import sys
import numpy as np
from flask import Response

caffe_root = '/home/ubuntu/Caffe/caffe-master/'
sys.path.insert(0,caffe_root + 'python')
import caffe

net = caffe.Classifier('./models/3-layered-caffenet/train_coolphabet_deploy.prototxt', './models/3-layered-caffenet/_iter_5000.caffemodel',mean=np.load('./models/3-layered-caffenet/out.npy').mean(1).mean(1),channel_swap=(2,1,0),raw_scale=255,image_dims=(227, 227))


app = Flask(__name__)

@app.route('/')
def api_root():
    return 'Welcome'

@app.route('/articles')
def api_articles():
    return 'List of ' + url_for('api_articles')

@app.route('/images/<path:img_name>')
def get_image_label(img_name):
    if img_name:
        #imgName = request.args['imgName']
        input_image = caffe.io.load_image('/home/ubuntu/coolpha_data/hpl-telugu-iso-char-offline-test/'+img_name)
        #print input_image
        prediction = net.predict([input_image])
        label = prediction[0].argmax()
        print list(prediction[0])
        data = {
        'img_name'  : img_name,
        'label' : str(label),
        'probabilities':[float(i) for i in list(prediction[0])]
        }
        js = json.dumps(data)
        resp = Response(js, status=200, mimetype='application/json')
        return resp
    else:
        message = {
            'status': 404,
            'message': 'Not Found: ' + request.url,
        }
        resp = jsonify(message)
        resp.status_code = 404
        return resp

@app.route('/articles/<articleid>')
def api_article(articleid):
    return 'You are reading ' + articleid

if __name__ == '__main__':
    app.run()
