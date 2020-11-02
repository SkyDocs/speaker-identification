# Speaker Identification

Speaker Identification using Neural Net.<br>

Run the file in [Google Colab](https://colab.research.google.com/drive/12lmdoBpwZkkrtI6jak9utgAYlpBEiEVM?usp=sharing).<br>

Epochs = 10<br>
Accuracy ~ 98%<br>
Speakers = 3<br>
Dataset time = 10 mins

This is a support repo for the main Project - [Personalised Voice Assistant](https://github.com/SkyDocs/personalised-voice-assistant).


## Data-Set

There are two options for the datasets. In either of the way you follow, download the dataset form [Kaggle](https://www.kaggle.com/kongaevans/speaker-recognition-dataset) and keep the `_background_noise_` &  `other` folder for the Noise.


### Kaggle

Dataset Source - [Kaggle](https://www.kaggle.com/kongaevans/speaker-recognition-dataset)<br>


### Self Made Data-sets

We recorded the speakers voice for 10 mins using the `record.py` file. And the obtained 10 mins `.wav` is then splitted into 1 seconds of wav files using `audio-clip.py`.

And then append the 1 second wav files into the data-set folder with the user name. 


## Usage

To use the Speaker Identification from scratch, you will be needing a data set. For the testing version of our model we trained on the *Speaker recognition dataset* from [kaggle](https://www.kaggle.com/kongaevans/speaker-recognition-dataset). <br>

I recommend making a fresh new environment for this project.

Install the requirements by running 

`pip install -r requirements.txt` or `pip3 install -r requirements.txt`

Download the Dataset or else create your own data set by speaking for 10 mins after running the file

`python record.py` or `python3 record.py`

And split the generated `data.wav` into 1 second files using the `audio-clip.py` file.<br>
Run 
`python audio-clip.py` or `python3 audio-clip.py`

And train the neural net by running the the `speaker-identification.ipynb` file or run

`python speaker_identification.py` or `python3 speaker_identification.py`

After training, save the generated `model.h5` locally in the root folder, the repo. And run

`python predict.py` or `python3 predict.py` 

(depending upon your environment) for real time predction.


## Contribution

This is a Purely open-source project, and feel free to suggest the changes.<br>

To contribute Fork it, make the changes and push a pull request and we will get back to you.


## License

Licensed under [GNU General Public License v3.0](https://github.com/SkyDocs/speaker-identification/blob/master/LICENSE)
