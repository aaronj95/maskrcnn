#Import packages
import json 
import numpy as np
import os
import tensorflow as tf
import pickle
import jsonpickle

import sys
import random
import math
import re
import time
import cv2

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

#Import Flask
from flask import Flask, request, Response
# Initialize the Flask application
app = Flask(__name__)

#Config
class InferenceConfig(Config):
    # Give the configuration a recognizable name
    NAME = "microcontroller_segmentation"
    
    NUM_CLASSES = 1 + 2

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    DETECTION_MIN_CONFIDENCE = 0.99

inference_config = InferenceConfig()

#Model
# Get path to saved weights
model_path = "logs/microcontroller_segmentation20201102T0729/mask_rcnn_microcontroller_segmentation_0003.h5"

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=model_path)

# Load trained weights
model.load_weights(model_path, by_name=True)


@app.route('/')
def hello_world():
  return 'Hello, World!'

@app.route('/predict_api',methods=['POST'])
def predict_api():
  score_class_names = ['BG', 'mysejahtera', 'phone']

  r = request
  # convert string of image data to uint8
  nparr = np.fromstring(r.data, np.uint8)
  # decode image
  score_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  #Score
  results = model.detect([score_image], verbose=1)
  r = results[0]
  arr = r['scores']
  
  # build a response dict to send back to client
  #response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])}
  # encode response using jsonpickle
  #response_pickled = jsonpickle.encode(response)
  if arr[1] >= 0.99:
    return Response(response=r, status=200, mimetype="application/json")


if __name__ == '__main__':
  app.run()
