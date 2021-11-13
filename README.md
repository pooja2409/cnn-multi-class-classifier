# PART-1:
## CNN-Multi-Class-Classifier usign Fashion MNIST Dataset

Model Architecture:
  The model has two main aspects: the feature extraction front end comprised of convolutional and pooling layers, and the classifier backend that will make a prediction. For the convolutional front-end, I started with a single convolutional layer with a small filter size (3,3) and a modest number of filters (32) followed by a max pooling layer. The filter maps can then be flattened to provide features to the classifier. Given that the problem is a multi-class classification, we know that it will require an output layer with 10 nodes in order to predict the probability distribution of an image belonging to each of the 10 classes. This will also require the use of a softmax activation function. Between the feature extractor and the output layer, I added a dense layer to interpret the features, in this case with 100 nodes. All layers will use the ReLU activation function and the weight initialization scheme, both best practices.
I used a conservative configuration for the stochastic gradient descent optimizer with a learning rate of 0.01 and a momentum of 0.9. The categorical cross-entropy loss function will be optimized, suitable for multi-class classification, and I will monitor the classification accuracy metric, which is appropriate given fashion mnist dataset has the same number of examples in each of the 10 classes.

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

# PART 2
## Unified API to send and receive messages to / from Apache Kafka and Google Pub/Sub.

### Setup Kafka: A distributed platform for reading and writing data streams ###
1. Download latest kafka tar.gz file
2. `cd kafka_2.12-3.0.0/`
3. Kafka requires ZooKeeper in order to run
`bin/zookeeper-server-start.sh config/zookeeper.properties`
4. Starting the Kafka server in default port 9092 
`bin/kafka-server-start.sh config/server.properties`
5. Now adding a topic in the subscription service 
`./kafka-topics.sh --create --topic my-topic --bootstrap-server localhost:9092 --replication-factor 1 --partitions 4` 
6. Kafka is set and reddy to be used

### Setup Google Pub-Sub Emulator: ###
The Pub/Sub Emulator for Kafka emulates the Pub/Sub API while using Kafka to process the messages. The emulator runs as a standalone Java application, which makes it easy to deploy alone or inside an AppScale deployment.
1. Cloning the kafka-pubsub-emulator repo in localhost
`git clone https://github.com/GoogleCloudPlatform/kafka-pubsub-emulator.git`
2. `cd kafka-pubsub-emulator`
3. Make sure you have java and mvn installed for the next step
4. `mvn clean package`
5. This will create a standalone jar file that we will be using in the next steps
6. The two files config.json and pubsub.json are created containing all the configs and subscription details, pass the path of the configs in the java -jar call
`java --add-opens java.base/java.lang=ALL-UNNAMED -jar target/kafka-pubsub-emulator-0.1.0.jar -c config.json -p pubsub.json`

### Unified API ###
Using the API in the application requires the Pub/Sub Client Libraries. Normally these libraries would contact the Cloud Pub/Sub service, but instead of pointing them to a cloud endpoint, I am pointing them to the local server.

Sender.py is used as a publisher that publishes a message and receiver.py has subscribed to the service and receives the message
To pulish a message use the following command
1. `cd kafka-pub-sub-unified-api`
2. Install the necessary libraries
`pip install -r requirements.txt`
3. Publish a message 'CNN Model is ready'
`python sender.py --host localhost --port 8080 --project-id test-project --topic-id test-project-8p "CNN Model is ready"`
4. This will publish the message and whichever service has subscribed to the same subscription, will be able to receive this message

To check if the receiver received this message, use the below command
1. `python receiver.py --host localhost --port 8080 --project-id test-project --subscription-id subscription-to-my-topic`
The output will look something like this
```
Message {
  data: b'CNN Model is ready'
  ordering_key: ''
  attributes: {}
}
```



