# PART 3
## Creating a pub-sub model for machine learning classifier
### Setting up the model server ###
The model server is created as a flask api that return the category of the prediction given the path of the input image

Steps to start the model server
1. activate the virtual env created in part 1
`source cnnenv.bin.activate`
2. Install other packages required for part 3
`pip install -r requirements.txt`
3. Move to the model_serving folder
`cd model_serving`
4. Run the model_server with arguments using model_server.py file
`python model_server.py --saved_model_location ../models/base_model.h5 --input_shape 28 28 1`
5. Please note it take two arguments the saved model location and the input shape that must have been used while training


Endpoints: <br />
    1. http://localhost:5000/ <br />
    The home page that displays a welcome message <br />
    2. http://localhost:5000/predict?image_path=../images/test1.png <br />
    This endpoint takes in the image path that needs to be classified, you can also give the complete path of the image <br />

### Setting up the Publisher to send async request to the model_server to make predictions ###
Created a publisher service that can be used as an endpoint to send images for prediction to the prediction model
The architechture is as suggested, the publisher reaches out with a request to the model and the model sends prediction to the message broker which inturn is received by the receiver(which is subscribed to the same topic)

Steps to start the sender service:
1. activate the virtual env created in part 1
`source cnnenv.bin.activate`
2. Install other packages required for part 3
`pip install -r requirements.txt`
3. Move to the model_serving folder
`cd model_serving`
4. Run the publisher_service with arguments using sender_service.py file
`python sender_service.py --host localhost --port 8080 --project-id test-project --topic-id test-project-8p --model_server_url http://localhost:5000/predict`
5. Please note for the sender service to function make sure that the model server is up and running.

Endpoint for sender service: <br />
    1. http://localhost:5001/ <br />
    The home page that displays a welcome message <br />
    2. http://localhost:5001/publish?image_path=../images/test1.png <br />
    This endpoint takes in the image path that needs to be classified, and send it to the model server, the response is then stored in the message broker

### Receiving the messages from the service ###
The receiver remains the same as in part 2, a queue that is subscribed to the same topic where the model results are published.
Run the receiver.py from part 2 to see the response
1. `cd kafka-pub-sub-unified-api` 
`python receiver.py --host localhost --port 8080 --project-id test-project --subscription-id subscription-to-my-topic`
The output will look something like this
```
Message {
  data: b'{  "result": {    "Image path": "../images/test1.png, "Predicted  Category": "sneaker"  }}'
  ordering_key: ''
  attributes: {}
}
Message {
  data: b'{  "result": {    "Image path": "../images/fashion0.png, "Predicted  Category": "sneaker"  }}'
  ordering_key: ''
  attributes: {}
}
Message {
  data: b'{  "result": {    "Image path": "../images/fashion1.png, "Predicted  Category": "t_shirt_top"  }}'
  ordering_key: ''
  attributes: {}
}

```
