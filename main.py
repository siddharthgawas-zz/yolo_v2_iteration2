# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:22:48 2019

@author: NH
"""

# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages

# from loss import yolo_loss
# from dataset import Dataset, ImageDataset
import numpy as np
# from network import get_custom_model, get_pre_trained_model
from config import Configuration
from non_max_supression import non_max_suppression, non_max_suppression2
# import matplotlib.pyplot as plt
import util
import cv2
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import flask
import io
import tensorflow as tf
from yolov2 import get_pre_trained_model_vgg16

tf.enable_eager_execution()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# initialize our Flask application and the Keras model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

IMAGE_H, IMAGE_W = Configuration.IMAGE_HEIGHT, Configuration.IMAGE_WIDTH
# N_CLASSES = len(Configuration.CLASS_LABELS.keys())
N_CLASSES = len(Configuration.CLASS_LABELS)
N_ANCHORS = Configuration.ANCHORS.shape[0]
ANCHORS = Configuration.ANCHORS
GRID_SIZE = Configuration.GRID_SIZE

app = flask.Flask(__name__)
model = None

def load_model():
	global model
	model = get_pre_trained_model_vgg16(IMAGE_W,
                                       IMAGE_H,
                                       GRID_SIZE,GRID_SIZE,N_ANCHORS,
                                       N_CLASSES,1,ANCHORS)
	model.load_weights('trained_weights/leaf_data_v2/yolo_net_epoch_102.h5')
	model._make_predict_function()
	model.summary()


def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")
	image = img_to_array(image)
	image = cv2.resize(image,(IMAGE_H,IMAGE_W))
	image = image / 255.0
	image = np.expand_dims(image, axis=0)


	# return the processed image
	return image

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	y_pred_1 = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))
			original_w,original_h = image.size
			

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(IMAGE_H, IMAGE_W))

			# classify the input image and then initialize the list
			# of predictions to return to the client
			y_pred = model.predict(image)
			y_pred_1 = y_pred[0]
			y_pred_1 = non_max_suppression(np.reshape(y_pred_1, (GRID_SIZE * GRID_SIZE * N_ANCHORS, 5 + N_CLASSES)), 0.4,
                                       0.2)
			result = []
			for i in range(y_pred_1.shape[0]):
				box = y_pred_1[i,0:4]
				box = np.array(util.cvt_coord_to_diagonal(box)).reshape(-1,2)
				box[:,0] = original_w*box[:,0]
				box[:,1] = original_h*box[:,1] 
				box = box.reshape(-1)
				box = np.ceil(box).astype(int)
				obj = {
						'box': box.tolist(),
						'infected': float(y_pred_1[i,5]),
						'not_infected': float(y_pred_1[i,6]),
						'objectness': float(y_pred_1[i,4])
						}
				result.append(obj)
	# return the data dictionary as a JSON response
	return flask.jsonify(result)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run(host="0.0.0.0")