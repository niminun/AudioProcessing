#
# A main script for running all stages of voice separation.
#
import os
from data_processing import audio_dir_to_spects
import argparse


SPECT_DIR = "spect"

if __name__ == '__main__':

    ######################## getting arguments ########################
    parser = argparse.ArgumentParser(description='Voice separation script.')
    parser.add_argument('--audio_dir', default=None,
                        help='a path to audio files directory. If provided, it '
                             'will be preprocessed for NN usage.')
    parser.add_argument('--out_dir', required=True,
                        help='a path to outputs dir.')
    parser.add_argument('--suffix', nargs='?', default=['mp3'],
                        help='all relevant audio files suffices')

    args = parser.parse_args()

    ######################## running operations ########################
    spect_dir = os.path.join(args.out_dir, SPECT_DIR)
    if not os.path.isdir(spect_dir):
        os.makedirs(spect_dir)

    # converting audio files to spectrogram
    if args.audio_dir:
        audio_dir_to_spects(args.audio_dir, spect_dir, args.suffix)

    # training a model TODO

    # generating separated voices TODO
