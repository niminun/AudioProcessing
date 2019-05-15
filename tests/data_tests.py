import os
import numpy as np
from glob import iglob
from matplotlib import pyplot as plt
from utils.display import plot_spectrogram

SPECT_H = 1025


def spects_to_img(spects_dir, out_dir, spect_height=SPECT_H, suffix=("bin",)):

    if not os.path.isdir(spects_dir):
        print("no such dir: {}".format(spects_dir))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for s in suffix:
        pattern = os.path.join(spects_dir, "*{}".format(s))
        for f_path in iglob(pattern):
            spect_name = os.path.splitext(os.path.basename(f_path))[0]
            out_path = os.path.join(out_dir, spect_name + ".png")
            spect = np.fromfile(f_path, dtype="float32").reshape(spect_height, -1)
            plot_spectrogram(spect, x_axis="time", y_axis="log", title=os.path.basename(spect_name), show=False)
            plt.savefig(out_path)
            plt.close()
            print(out_path)
