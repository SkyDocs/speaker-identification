import sys
import tensorflow as tf

if len(sys.argv) == 1 or len(sys.argv) > 2:
	print("Usage : pass the saved keras model path")
	sys.exit()


if len(sys.argv) == 2:
	saved_model_dir = sys.argv[1]

print("\033[31m[*] Converting \033[0m ")

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  	f.write(tflite_model)

print("\033[31m[*] Finished the convertion. File is saved as model.tflite \033[0m ")