from data_processing import *
import matplotlib.pyplot as plt
import matplotlib.style as ms
import librosa.display

ms.use('seaborn-muted')


def plot_spectrogram(spect, x_axis, y_axis, sr=SR, title='', show=True):
    h, w = spect.shape
    plt.figure(figsize=(int(w/100 + 0.99), int(h/100 + 0.99)))

    # plot
    librosa.display.specshow(spect, sr=sr, x_axis=x_axis, y_axis=y_axis)

    # Put a descriptive title on the plot
    plt.title(title)

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    if show:
        plt.show()
