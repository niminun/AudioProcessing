#
# A main script for running all stages of voice separation.
#
import os
from data_processing import audio_dir_to_spects
from tests.data_tests import spects_to_img

################################################ preprocess arguments ##################################################
CONVERT_AUDIO = True
AUDIO_DIR = r"C:\\Users\nimro\Music\VoiceSeparation\NoDrums"
OUT_DIR = r"C:\\Users\nimro\Projects\VoiceSeparation\V1\no_drums"
SPECT_DIR = "spect"
FIG_DIR = "figures"
SUFFIX = ["mp3", "wav", "wma"]

################################################# training arguments ###################################################
DO_TRAIN = False
BASE_DIR = r"C:\\Users\nimro\Projects\VoiceSeparation\V1\model_training"
ALL_INSTRUMENTS_DATA = ""
DRUMS_DATA = ""
NO_DRUMS_DATA = ""

########################################################################################################################
if __name__ == '__main__':

    spect_dir = os.path.join(OUT_DIR, SPECT_DIR)
    figs_dir = os.path.join(OUT_DIR, FIG_DIR)
    for d in [spect_dir, figs_dir]:
        if not os.path.isdir(d):
            os.makedirs(d)

    ######################## running operations ########################

    # converting audio files to spectrogram
    if CONVERT_AUDIO:
        # converting audio files to spectrogram
        audio_dir_to_spects(AUDIO_DIR, spect_dir, SUFFIX)
        # saving spectrogram figures
        spects_to_img(spect_dir, figs_dir)

    # training separator
    if DO_TRAIN:
        command = "model_trainer.py --base_dir={} --x_data_dir={}, --y_data_dir={} --z_data_dir{}" \
            .format(BASE_DIR, DRUMS_DATA, NO_DRUMS_DATA, ALL_INSTRUMENTS_DATA)

    # generating separated voices TODO
