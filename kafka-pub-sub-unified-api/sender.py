import argparse
import os

# libarary to connect to google pub-sub
import grpc
from google.cloud import pubsub_v1

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
    	description='Publishes messages to a topic')
	parser.add_argument('--host', required=True, help='The emulator host or IP address')
	parser.add_argument('--port', type=int, required=True,
                    	help='The emulator port number')
	parser.add_argument('--project-id', required=True)
	parser.add_argument('--topic-id', required=True)
	parser.add_argument('message')
	args = parser.parse_args()

	emulator_location = ':'.join([args.host, str(args.port)])
	# setting the emulator path that will be used by pubsub_v1
	os.environ['PUBSUB_EMULATOR_HOST'] = emulator_location
	publisher = pubsub_v1.PublisherClient()

	# setting projectid and topic id
	topic_path = publisher.topic_path(args.project_id, args.topic_id)
	# publishing the message in bytes format
	publisher.publish(topic_path, bytes(args.message,encoding='utf8')).result()