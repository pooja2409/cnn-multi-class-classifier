# PART-1:
## CNN-Multi-Class-Classifier usign Fashion MNIST Dataset

Model Architecture:
  The baseline CNN model consists of 1 convolution layer, 1 maxpool layer,  and 2 dense layers.
  RELU activation is used to add some non-linearity in the model and 
  Stochastic gradient decent over normal gradient decent is used for faster calculations than gradient descent and batch gradient descent
  The baseline model can be improved by making the network more robus by adding more CONV2D layers. This will improve the feature generation and
  to make sure the model doesnot overfit on the training data drop outs can be added in between during training.
  

Setup the environment to train/test the model
1. Create a virtual environment:
`python -m venv cnnenv`
2. Activate the virtual environment:
`source cnnenv/bin/activate`
3. Install all the necessary packages listed in requirements.txt to train and test the model:
`pip install -r requirements.txt`
4. Once the packages are installed, you are all set to start training the model

Steps to train the model:
1.  If not already active, Activate the virtual environment:
`source cnnenv/bin/activate`
2. The model can be trained using train.py file which takes multiple arguments that can be used to tune the parameters.

Case - 1: Run the below command to train the model using Fashion MNIST dataset
```
python train.py \
	--dataset fmnist \
	--num_classes 10 \
	--input_shape 28 28 1 \
	--lr 0.01 \
	--momentum 0.9 \
	--epochs 10 \
	--batch_size 32 \
	--verbose 1 \
	--n_folds 5 \
	--saved_model_location models/base_model.h5
```
1. The above command sets the dataset to fmnist data which has 10 classes and the input shape of the images is set to be (28,28,1), hyperparameters such as learning rate(lr), momentum, epochs, n_folds
can be set using the arguments.
2. The above command will train the model and store the trained model in the location models/base_model.h5

Case - 2: If you have your own custom data, you can load that data as well for training using the arguments.
The data shpuld be stored in data folder in the structure mentioned below:
-- data
  -- class1
    -- img1.png
    -- img2.png
   -- class2
    -- img1.png
    -- img2.png
 
1. Please note the folder names class1, class2 will be the name of the classes(labels)
2. You can use the below command to train your own custom data:
```
python train.py \
	--dataset other \
	--num_classes 10 \
	--input_shape 28 28 1 \
	--lr 0.01 \
	--momentum 0.9 \
	--epochs 10 \
	--batch_size 32 \
	--verbose 1 \
	--n_folds 5 \
	--saved_model_location models/base_model_custom_dataset.h5 \
	--dataset_location data/
```

Steps to test the model:
1. Once the training is finished, you should be able to see the trained model inside the models folder
2. To test/evaluate the model with testset of Fashion MNIST dataset, Please use the below command
```
python predict.py \
	--input_shape 28 28 1 \
	--verbose 1 \
	--saved_model_location models/base_model.h5 \
```
3. This will return the accuracy of the model on the testset of Fashion MNIST dataset
4. If you want to see the model's prediction on a custom image, pass the location of the image using arguments and the model will show the prediction class.
5. If the model is not trained on fashion mnist dataset you can also pass the labels as argument.
```
python predict.py \
	--input_shape 28 28 1 \
	--verbose 1 \
	--saved_model_location models/base_model.h5 \
    --use_image \
    --image_location images/test1.png \
	--custom_labels 't_shirt_top' 'trouser' 'pullover' 'dress' 'coat' 'sandal' \
			'shirt' 'sneaker' 'bag' 'ankle_boots'
```

** Please note there is .sh file present in the repo for both training and prediction that contains commands to run the model



