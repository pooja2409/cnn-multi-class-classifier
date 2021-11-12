# baseline cnn model for fashion mnist
import argparse
import os
import cv2
import numpy as np

from sklearn.model_selection import KFold
from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

# adding tensorflow CPU"s extention
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# parse hyperparameters used for training as arguments
def parse_arguments():
	parser = argparse.ArgumentParser(description="Process some integers.")
	parser.add_argument("--dataset", type=str, default='fmnist', help="NName of the dataset to use, fmnist for Fashion Mnist data")
	parser.add_argument("--num_classes", type=int, default=10, help="Number of classes for multi-class classification, 10 for fmnist data")
	parser.add_argument("--input_shape", nargs="+", type=int, default=[28, 28, 1], help="Shape of the input image")
	parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for CNN model")
	parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for CNN model")
	parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train the model")
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size to be used while training")
	parser.add_argument("--verbose", type=int, default=1, help="Verbose for model, 0 for silent")
	parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for K fold Validation")
	parser.add_argument("--saved_model_location", type=str, default="models/", help="Location to save the model after training")
	parser.add_argument("--dataset_location", type=str, default="data/", help="Location where image data is stored")
	return parser.parse_args()
 
# load Fashion MNIST train and test dataset
def load_fmnist_dataset(input_shape):
	# load dataset
	(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], input_shape[0], input_shape[1], input_shape[2]))
	testX = testX.reshape((testX.shape[0], input_shape[0], input_shape[1], input_shape[2]))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# load train and test dataset using image and label files
def load_generic_dataset(img_folder, input_shape):   
    img_data_array=[]
    class_name=[]
   
   # iterate over the directory
    for dir1 in os.listdir(img_folder):
		# iterate over each image file in dir
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path= os.path.join(img_folder, dir1,  file)
			# read the image using cv2
            image= cv2.imread(image_path, cv2.COLOR_BGR2RGB)
			# resize image based on input shape
            image=cv2.resize(image, (input_shape[0], input_shape[1]),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name

 
# define cnn model
def define_model(input_shape):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", input_shape=input_shape))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation="relu", kernel_initializer="he_uniform"))
	model.add(Dense(10, activation="softmax"))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
	return model
 
# train a model using k-fold cross-validation
def train_model(dataX, dataY, args):
	# prepare cross validation
    kfold = KFold(args.n_folds, shuffle=True, random_state=1)
	# enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
		# define model
        model = define_model(tuple(args.input_shape))
		# select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        model.fit(trainX, trainY, epochs=args.epochs, batch_size=args.batch_size, validation_data=(testX, testY), verbose=args.verbose)
    return model


# function to save the model once the training is complete
def save_model(model, saved_model_location):
    model.save(saved_model_location)


# driver function to train the model with arguments
def main():
    args = parse_arguments()
    print("Starting the process to load the fmnist dataset")
    if args.dataset == "fmnist":
        trainX, trainY, _, _ = load_fmnist_dataset(args.input_shape)
    else:
        trainX, trainY = load_generic_dataset(args.dataset_location, args.input_shape)
    print("Dataset Loaded")
    print("Starting Training")
    model = train_model(trainX, trainY, args)
    print(f"Saving model in folder {args.saved_model_location}")
    save_model(model, args.saved_model_location)

main()


