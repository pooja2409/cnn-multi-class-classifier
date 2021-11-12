# baseline cnn model for fashion mnist
import argparse
import imageio
import os
import numpy as np
from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from keras.models import load_model

# adding tensorflow CPU"s extention
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# parse hyperparameters used for training as arguments
def parse_arguments():
	parser = argparse.ArgumentParser(description="Process some integers.")
	parser.add_argument("--input_shape", nargs="+", type=int, default=[28, 28, 1], help="Shape of the input image")
	parser.add_argument("--verbose", type=int, default=1, help="Verbose for model, 0 for silent")
	parser.add_argument("--saved_model_location", type=str, default="models/base_model.h5", help="Location to save the model after training")
	parser.add_argument("--use_image", default=False, action='store_true', help="use a new image for testing outside fmnist dataset")
	parser.add_argument("--image_location", type=str, default="images/test1.png", help="image for testing outside fmnist dataset")
	parser.add_argument("--custom_labels", nargs="+", type=str, default= \
		['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', \
			'shirt', 'sneaker', 'bag', 'ankle_boots'], help="labels for testing outside fmnist dataset")
	
	return parser.parse_args()

 
# load train and test dataset
def load_dataset(input_shape):
	# load dataset
	(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], input_shape[0], input_shape[1], input_shape[2]))
	testX = testX.reshape((testX.shape[0], input_shape[0], input_shape[1], input_shape[2]))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
 
 
# evaluate a model using k-fold cross-validation
def evaluate_model(testX, testY, verbose, saved_model_location):
    model = load_model(saved_model_location)
    _, acc = model.evaluate(testX, testY, verbose=1)
    return round(acc * 100.0, 2)


# predict the output on a single image
def predict(img, saved_model_location, input_shape):
	img = np.resize(img, input_shape)
	model = load_model(saved_model_location)
	return np.argmax(model.predict(img.reshape(1, input_shape[0], input_shape[1], input_shape[2]))[0])


# driver function to test the model with arguments
def main():
	args = parse_arguments()
	if not args.use_image:
		print("Evaluating model on Fashion MNIST dataset")
		_, _, testX, testY= load_dataset(args.input_shape)
		acc = evaluate_model(testX, testY, args.verbose, args.saved_model_location)
		print(f"Accuracy of the model is {acc}%")
	else:
		print("Predicting output using the image file")
		img = imageio.imread(args.image_location)
		result = predict(img, args.saved_model_location, tuple(args.input_shape))
		print(f"The prediction for given input image is : {args.custom_labels[result]}")

main()
