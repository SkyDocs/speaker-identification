import os
import sys
import tensorflow as tf
from zipfile import ZipFile

if len(sys.argv) == 1 or len(sys.argv) > 2:
	print("Usage : pass the saved keras model path (model_keras_tflite.zip)")
	sys.exit()


if len(sys.argv) == 2:
	saved_model = sys.argv[1]

print("\033[31m[*] Converting \033[0m ")

with ZipFile('saved_model.zip', 'r') as zipObj:
	zipObj.extractall()

converter = tf.lite.TFLiteConverter.from_saved_model(os.getcwd())
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  	f.write(tflite_model)

print("\033[31m[*] Finished the convertion. File is saved as model.tflite \033[0m ")