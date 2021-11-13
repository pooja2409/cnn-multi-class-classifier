import argparse
import os
import requests
import flask
from flask import request,jsonify, make_response

# libarary to connect to google pub-sub
import grpc
from google.cloud import pubsub_v1
parser = argparse.ArgumentParser(
    description='Publishes messages to a topic')
parser.add_argument('--host', required=True, help='The emulator host or IP address')
parser.add_argument('--port', type=int, required=True,
                    help='The emulator port number')
parser.add_argument('--project-id', required=True)
parser.add_argument('--topic-id', required=True)
parser.add_argument('--model_server_url', required=True)
args = parser.parse_args()

emulator_location = ':'.join([args.host, str(args.port)])
# setting the emulator path that will be used by pubsub_v1
os.environ['PUBSUB_EMULATOR_HOST'] = emulator_location
publisher = pubsub_v1.PublisherClient()

# setting projectid and topic id
topic_path = publisher.topic_path(args.project_id, args.topic_id)
# publishing the message in bytes format

app = flask.Flask(__name__)


@app.route('/publish')
def publish_results():
    if request.args.get('image_path') is not None:
        image_path = request.args.get('image_path')
        # get prediction
        resp = requests.get(args.model_server_url + "?image_path=" + image_path)
        if resp.status_code == 200:
            publisher.publish(topic_path, bytes(resp.text.replace("\n", ""),encoding='utf8')).result()
            print("response published")
            return make_response(jsonify(resp.text), 200)
        else:
            publisher.publish(topic_path, bytes("Error! Prediction cannot be made, please check \
                if the model path and image path is valid",encoding='utf8')).result()
            print("response published")
            return "Error! Prediction cannot be made, please check if the model path and image path is valid"
    else:
        return make_response('Parameter image_path not found! Please provide a valid image_path to check its predicted category', 400)


@app.route('/')
def home():
    """
    returns homepage
    """
    return("Welcome to the publisher service for fashion mnist dataset prediction")


if __name__ == '__main__':
	app.run(use_reloader=True, port=5001, debug=True,host='0.0.0.0')