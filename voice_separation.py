#
# A main script for running all stages of voice separation.
#
import os
from data_processing import audio_dir_to_spects

CONVERT_AUDIO = False
AUDIO_DIR = ""
SPECT_DIR = "spect"
OUT_DIR = r"C:\\Users\nimro\Projects\VoiceSeparation\V1\spects"
SUFFIX = ["mp3"]

BASE_DIR = r"C:\\Users\nimro\Projects\VoiceSeparation\V1\model_training"
ALL_INSTRUMENTS_DATA = ""
DRUMS_DATA = ""
NO_DRUMS_DATA = ""

if __name__ == '__main__':

    spect_dir = os.path.join(OUT_DIR, SPECT_DIR)
    if not os.path.isdir(spect_dir):
        os.makedirs(spect_dir)

    ######################## running operations ########################

    # converting audio files to spectrogram
    if CONVERT_AUDIO:
        audio_dir_to_spects(AUDIO_DIR, spect_dir, SUFFIX)

    # training separator
    command = "model_trainer.py --base_dir={} --x_data_dir={}, --y_data_dir={} --z_data_dir{}" \
              .format(BASE_DIR, DRUMS_DATA, NO_DRUMS_DATA, ALL_INSTRUMENTS_DATA)

    # generating separated voices TODO
