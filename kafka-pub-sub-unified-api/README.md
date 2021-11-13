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
6. The two files config.json and pubsub.json are created containing all the configs and subscription details, use these files in the java -jar call
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


