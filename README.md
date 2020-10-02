# Speaker Identification

Speaker Identification using Neural Net.<br>

Run the file in [Google Colab](https://colab.research.google.com/drive/12lmdoBpwZkkrtI6jak9utgAYlpBEiEVM?usp=sharing).<br>

This is a support repo for the main Project - [Personalised Voice Assistant](https://github.com/SkyDocs/personalised-voice-assistant).

Dataset Source - [Kaggle](https://www.kaggle.com/kongaevans/speaker-recognition-dataset)<br>
Epochs = 10<br>
Accuracy ~ 90%<br>


## Usage

To use the Speaker Identification from scratch, you will be needing a data set. For the testing version of our model we trained on the *Speaker recognition dataset* from [kaggle](https://www.kaggle.com/kongaevans/speaker-recognition-dataset). <br>

Install the requirements by running 

`pip install -r requirements.txt` or `pip3 install -r requirements.txt`

And train the neural net by running the the `speaker-identification.ipynb` file or run

`python speaker_identification.py` or `python3 speaker_identification.py`

After training, save the generated `model.h5` locally in the root folder, the repo. And run

`python predict.py` or `python3 predict.py` 

(depending upon your environment) for real time predction.


## Contribution

This is a Purely open-source project, and feel free to suggest the changes.<br>

To contribute Fork it, make the changes and push a pull request and we will get back to you.


## License

Licensed under [MIT License](https://github.com/SkyDocs/personalised-voice-assistant/blob/master/LICENSE)