# Speaker Identification

Speaker Identification using Neural Net.<br>

Run the file in [Google Colab](https://colab.research.google.com/github/SkyDocs/speaker-identification/blob/master/speaker-identification.ipynb)<br>

Run the Beta version in [Google Colab](https://colab.research.google.com/drive/12lmdoBpwZkkrtI6jak9utgAYlpBEiEVM#scrollTo=5j0lgVmAC7-V)<br>

Epochs = 10<br>
Accuracy ~ 98%<br>
Speakers = 3<br>
Dataset time = 10 mins

This is a support repo for the main Project - [Personalised Voice Assistant](https://github.com/SkyDocs/personalised-voice-assistant).

***Make sure that you are using the same tensorflow version while trainging and running. Otherwise it will throw error `ValueError: Unknown layer: Functional`***

***ref: [ValueError: Unknown layer: Functional](https://stackoverflow.com/questions/63068639/valueerror-unknown-layer-functional)***

## Data-Set

There are two options for the datasets. In either of the way you follow, download the dataset form [Kaggle](https://www.kaggle.com/kongaevans/speaker-recognition-dataset) and keep the `_background_noise_` &  `other` folder for the Noise.

And pass the noise folder in the noise path while training and the same in the identification one too. 


### Kaggle

Dataset Source - [Kaggle](https://www.kaggle.com/kongaevans/speaker-recognition-dataset)<br>


### Self Made Data-sets

We recorded the speakers voice for 10 mins using the `record.py` file. And the obtained 10 mins `.wav` is then splitted into 1 seconds of wav files using `audio_clip.py`.

And then append the 1 second wav files into the data-set folder with the user name. 


## Usage

To use the Speaker Identification from scratch, you will be needing a data set. For the testing the our model we initially trained on the *Speaker recognition dataset* from [kaggle](https://www.kaggle.com/kongaevans/speaker-recognition-dataset). Then we created our own dataset for the recognition purpose.<br>
### Dataset details:
- time duration: 10 mins
- splitted into 1 sec using audio_clip.py

### Training
I recommend making a fresh new environment for this project.

Install the requirements by running 

`pip install -r requirements.txt` or `pip3 install -r requirements.txt`

Download the Dataset or else create your own data set by speaking for 10 mins after running the file

`python record.py` or `python3 record.py`

And split the generated `data.wav` into 1 second files using the `audio_clip.py` file.<br>
Run 
`python audio_clip.py` or `python3 audio_clip.py`

And train the neural net by running the the `speaker-identification.ipynb` file or run

`python speaker_identification.py` or `python3 speaker_identification.py`

### For Regular devices
*(devices with at least i3 processor or equivalent and 4 gigs of RAM)*

After training, you will get the generated `model.h5` locally in the root folder of the repo. 
And run

`python predict.py` or `python3 predict.py` 

(depending upon your environment) for real time prediction. And then speak when prompted and let he

### For arm/edge devices
*(devices like raspberry-pi & mobile devices(Android & iOS))*

For these devices you can optimize the model for them using the script `tflite_convert.py`, but the *accuracy may reduce*.

After successful training of the model you will also get `model_keras_tflite.zip` in your root folder of the repo.

For the conversion of the standard keras model run 

`python tflite_convert.py path_to_model_keras_tflite.zip` or `python3 tflite_convert.py path_to_model_keras_tflite.zip`

And you will get a `model.tflite` for your arm/edge devices!

## Contribution

This is a Purely open source project, and feel free to suggest changes.<br>

To contribute, Fork it, make the changes and push a pull request and we will get back to you.

## Reporting a bug

Create a issue, explaining how to reproduce the issue.

## Disclaimer

We provide limited support for the tflite models.
 

## License

Licensed under [GNU General Public License v3.0](https://github.com/SkyDocs/speaker-identification/blob/master/LICENSE)
