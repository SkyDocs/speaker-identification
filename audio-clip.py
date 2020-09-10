# For clipping a wav file into multiple 1 second wav files for the self made datasets

# pip install pydub

from pydub import AudioSegment
import math
import os

class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '/' + filename
        
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        # print(self.audio.duration_seconds)
        return self.audio.duration_seconds
    
    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 1000
        t2 = to_min * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '/' + split_filename, format="wav")
        
    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration())
        for i in range(0, total_mins, min_per_split):
            split_fn = str(i) + '.wav'
            self.single_split(i, i+min_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_mins - min_per_split:
                print('All splited successfully')


"""
# if you are having .mp3 file only
# mp3 -> .wav
"""
# sound = AudioSegment.from_mp3("/path/to/file.mp3")
# sound.export("/output/path/file.wav", format="wav")


folder = ''  # wav files folder
file = '' # .wav file
split_wav = SplitWavAudioMubin(folder, file)
split_wav.multiple_split(min_per_split=1)

"""
Optional: Remove the file which is splitted
"""
# os.remove(folder + '/' + file)