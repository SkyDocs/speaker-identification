import os
clear = lambda: os.system('clear')
clear()

print("\033[31m[*]\033[0m You will be asked to speak for few seconds for the recognition of the speaker.")

import time
import shutil
import numpy as np
import pyaudio
import wave
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from IPython.display import display, Audio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATASET_ROOT = "/Users/harshitruwali/Desktop/sem3-pro/"
NOISE_SUBFOLDER = "noise"
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)

SAMPLING_RATE = 16000
SHUFFLE_SEED = 43
BATCH_SIZE = 128
SCALE = 0.5

print("\033[31m[*]\033[0m Get Ready!")

time.sleep(5)

""" Taking the voice input """

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 16000  # Record at 16000 samples per second
seconds = 3
filename = "predict.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

# print("-------------------------------------------------------------------------------------------")
print("\033[31m[*]\033[0m Recording")

stream = p.open(format=sample_format,
				channels=channels,
				rate=fs,
				frames_per_buffer=chunk,
				input=True)

frames = []  # Initialize array to store frames

# Store data in chunks for 1 seconds
for i in range(0, int(fs / chunk * seconds)):
	data = stream.read(chunk)
	frames.append(data)

# Stop and close the stream
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print("\033[31m[*]\033[0m Finished recording")
# print("-------------------------------------------------------------------------------------------")
# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

print("\033[31m[*]\033[0m Processing")
"""Pre-processing Noise"""

# If folder noise, does not exist, create it, otherwise do nothing
if os.path.exists(DATASET_NOISE_PATH) is False:
	os.makedirs(DATASET_NOISE_PATH)

for folder in os.listdir(DATASET_ROOT):
	if os.path.isdir(os.path.join(DATASET_ROOT, folder)):
		if folder in [NOISE_SUBFOLDER]:
			# If folder is audio or noise, do nothing
			continue
		elif folder in ["other", "_background_noise_"]:
			# If folder is one of the folders that contains noise samples move it to the noise folder
			shutil.move(
				os.path.join(DATASET_ROOT, folder),
				os.path.join(DATASET_NOISE_PATH, folder),
			)
		else:
			pass

"""Noise"""

# Get the list of all noise files
noise_paths = []
for subdir in os.listdir(DATASET_NOISE_PATH):
	subdir_path = Path(DATASET_NOISE_PATH) / subdir
	if os.path.isdir(subdir_path):
		noise_paths += [
			os.path.join(subdir_path, filepath)
			for filepath in os.listdir(subdir_path)
			if filepath.endswith(".wav")
		]

# print("Found {} files belonging to {} directories".format(len(noise_paths), len(os.listdir(DATASET_NOISE_PATH))))

command = (
	"for dir in `ls -1 " + DATASET_NOISE_PATH + "`; do "
	"for file in `ls -1 " + DATASET_NOISE_PATH + "/$dir/*.wav`; do "
	"sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
	"$file | grep sample_rate | cut -f2 -d=`; "
	"if [ $sample_rate -ne 16000 ]; then "
	"ffmpeg -hide_banner -loglevel panic -y "
	"-i $file -ar 16000 temp.wav; "
	"mv temp.wav $file; "
	"fi; done; done"
)
os.system(command)

# Split noise into chunks of 16,000 steps each
def load_noise_sample(path):
	sample, sampling_rate = tf.audio.decode_wav(
		tf.io.read_file(path), desired_channels=1
	)
	if sampling_rate == SAMPLING_RATE:
		# Number of slices of 16000 each that can be generated from the noise sample
		slices = int(sample.shape[0] / SAMPLING_RATE)
		sample = tf.split(sample[: slices * SAMPLING_RATE], slices)
		return sample
	else:
		print("Sampling rate for {} is incorrect. Ignoring it".format(path))
		return None


noises = []
for path in noise_paths:
	sample = load_noise_sample(path)
	if sample:
		noises.extend(sample)
noises = tf.stack(noises)

# print(
# 	"{} noise files were split into {} noise samples where each is {} sec. long".format(
# 		len(noise_paths), noises.shape[0], noises.shape[1] // SAMPLING_RATE
# 	)
# )

def paths_and_labels_to_dataset(audio_paths, labels):
	"""Constructs a dataset of audios and labels."""
	path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
	audio_ds = path_ds.map(lambda x: path_to_audio(x))
	label_ds = tf.data.Dataset.from_tensor_slices(labels)
	return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
	"""Reads and decodes an audio file."""
	audio = tf.io.read_file(path)
	audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
	return audio


def add_noise(audio, noises=None, scale=0.5):
	if noises is not None:
		# Create a random tensor of the same size as audio ranging from
		# 0 to the number of noise stream samples that we have.
		tf_rnd = tf.random.uniform(
			(tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
		)
		noise = tf.gather(noises, tf_rnd, axis=0)

		# Get the amplitude proportion between the audio and the noise
		prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
		prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

		# Adding the rescaled noise to audio
		audio = audio + noise * prop * scale

	return audio


def audio_to_fft(audio):
	# Since tf.signal.fft applies FFT on the innermost dimension,
	# we need to squeeze the dimensions and then expand them again
	# after FFT
	audio = tf.squeeze(audio, axis=-1)
	fft = tf.signal.fft(
		tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
	)
	fft = tf.expand_dims(fft, axis=-1)

	# Return the absolute value of the first half of the FFT
	# which represents the positive frequencies
	return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])



def predict(path, labels):
	test = paths_and_labels_to_dataset(path, labels)


	test = test.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
	BATCH_SIZE
	)
	test = test.prefetch(tf.data.experimental.AUTOTUNE)


	test = test.map(lambda x, y: (add_noise(x, noises, scale=SCALE), y))

	for audios, labels in test.take(1):
		# Get the signal FFT
		ffts = audio_to_fft(audios)
		# Predict
		y_pred = model.predict(ffts)
		# Take random samples
		rnd = np.random.randint(0, 1, 1)
		audios = audios.numpy()[rnd, :]
		labels = labels.numpy()[rnd]
		y_pred = np.argmax(y_pred, axis=-1)[rnd]

		print("\033[31m[*]\033[0m Prediction")

		for index in range(1):
			print(
				"\033[31m[*]\033[0m Predicted:\33{} {}\33[0m".format(
					"[92m",y_pred[index]
				)
			)
			if y_pred[index] == 0:
				print("\033[31m[*]\033[0m Welcome user 0")
			elif y_pred[index] == 1:
				print("\033[31m[*]\033[0m Welcome user 1")
			elif y_pred[index] == 2:
				print("\033[31m[*]\033[0m Welcome user 2")
			elif y_pred[index] == 3:
				print("\033[31m[*]\033[0m Welcome user 3")
			else:
				print("\033[31m[*]\033[0m Welcome new user")


""" Predict """
path = ["predict.wav"]
labels = ["unknown"]
model = tf.keras.models.load_model('model.h5')
predict(path, labels)