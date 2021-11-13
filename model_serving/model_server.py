import flask
import argparse
import os
import numpy as np
import imageio
import warnings
warnings.filterwarnings("ignore")

from keras.models import load_model
from flask import request,jsonify, make_response

app = flask.Flask(__name__)

parser = argparse.ArgumentParser(
    	description='Publishes messages to a topic')
parser.add_argument('--saved_model_location', default='../models/base_model.h5', help='Location of the saved model')
parser.add_argument("--input_shape", nargs="+", type=int, default=[28, 28, 1], help="Shape of the input image")
parser.add_argument("--custom_labels", nargs="+", type=str, default= \
		['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', \
			'shirt', 'sneaker', 'bag', 'ankle_boots'], help="labels for testing outside fmnist dataset")
args = parser.parse_args()


# loading pre trained base model for fashion mnist dataset
def load_pretrained_model():
    model_location = args.saved_model_location
    if os.path.isfile(model_location):
        model = load_model(model_location)
        return model
    else:
        print(f"Model Not found in location {model_location}")
        exit()

model = load_pretrained_model()

# predict the output on a single image
def predict_category(img, input_shape):
	img = np.resize(img, input_shape)
	return np.argmax(model.predict(img.reshape(1, input_shape[0], input_shape[1], input_shape[2]))[0])


# endpoint to drive the prediction given image_path as argument in the url
@app.route('/predict')
def predict():
    """
    method that returns a json response of prediction of category given image_path
    """
    if request.args.get('image_path') is not None:
        image_path = request.args.get('image_path')
        # get prediction
        img = imageio.imread(image_path)
        result = predict_category(img, tuple(args.input_shape))
        predictedLabel = args.custom_labels[result]
        output = {'result':{'Image path':image_path, 'Predicted  Category': predictedLabel}}
        return make_response(jsonify(output), 200)
        
    else:
        return make_response('Parameter image_path not found! Please provide a valid image_path to check its predicted category', 400)


# function to display homepage
@app.route('/')
def home():
    """
    returns homepage
    """
    return("Welcome to the model server for fashion mnist dataset prediction")

if __name__ == '__main__':
    app.run(use_reloader=True, debug=True,host='0.0.0.0')

