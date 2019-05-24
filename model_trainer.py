from torch.utils.data import DataLoader
from data_processing import SpectrogramDataset
import torch
import numpy as np

SAMPLE_W = 64
SAMPLE_H = 1025
HIST_LEN = 256

TRUE_LABEL = 1
FALSE_LABEL = -1

# CUDA device check
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class VoiceSeparationTrainer(object):
    """
    A class for the purpose of training all required models for voice separation.
    """

    def __init__(self, generator, discriminator_x, discriminator_y, sample_height=SAMPLE_H,
                 sample_width=SAMPLE_W, history_len=HIST_LEN):
        """
        :param generator: a model that generates Y distributed data given Z=(X+Y) distributed data.
        :param discriminator_x: a model that discriminates between generated X data and real X data.
        :param discriminator_y: a model that discriminates between generated Y data and real Y data.
        :param sample_height: (optional) height of input sample.
        :param sample_width: (optional) width of input sample.
        :param history_len: (optional) size of history buffer.
        """
        self._gen = generator
        self._disc_x = discriminator_x
        self._disc_y = discriminator_y
        self._sample_height = sample_height
        self._sample_width = sample_width
        self._history_len = history_len
        self._gen_x_history_buffer = \
            torch.zeros((self._history_len, 1, self._sample_height, self._sample_width)).to(device)
        self._gen_y_history_buffer = \
            torch.zeros((self._history_len, 1, self._sample_height, self._sample_width)).to(device)
        self._x_data_loader = None
        self._y_data_loader = None
        self._z_data_loader = None

    def train(self, x_data_dir, y_data_dir, z_data_dir, max_iters=1000, gen_steps_per_iter=10,
              x_disc_steps_per_iter=10, y_disc_steps_per_iter=10, eval_interval=None, batch_size=32,):
        """
        Training the models
        :param x_data_dir: A directory containing X data.
        :param y_data_dir: A directory containing Y data.
        :param z_data_dir: A directory containing Z data.
        :param max_iters: (optional) number of iterations to run.
        :param gen_steps_per_iter: (optional) number of generator training steps to run in each iteration.
        :param x_disc_steps_per_iter: (optional) number of x discriminator training steps to run in each iteration.
        :param y_disc_steps_per_iter: (optional) number of y discriminator training steps to run in each iteration.
        :param eval_interval: (optional) number of iterations between each models evaluation.
        :param batch_size: size of batches to be used for each training iteration of all models.
        """

        eval_interval = eval_interval if eval_interval else int(max_iters / 100)  # do 100 evaluations by default.

        # creating data generators
        x_data_set = SpectrogramDataset(in_dir=x_data_dir, suffix="__mag_db.bin", sample_width=self._sample_width,
                                        spect_h=self._sample_height, label=TRUE_LABEL)
        y_data_set = SpectrogramDataset(in_dir=y_data_dir, suffix="__mag_db.bin", sample_width=self._sample_width,
                                        spect_h=self._sample_height, label=TRUE_LABEL)
        z_data_set = SpectrogramDataset(in_dir=z_data_dir, suffix="__mag_db.bin", sample_width=self._sample_width,
                                        spect_h=self._sample_height, label=TRUE_LABEL)
        self._x_data_loader = InfiniteDataLoader(DataLoader(dataset=x_data_set, batch_size=int(batch_size/2),
                                                            shuffle=True, num_workers=2, drop_last=True))
        self._y_data_loader = InfiniteDataLoader(DataLoader(dataset=y_data_set, batch_size=int(batch_size/2),
                                                            shuffle=True, num_workers=2, drop_last=True))
        self._z_data_loader = InfiniteDataLoader(DataLoader(dataset=z_data_set, batch_size=int(batch_size/2),
                                                            shuffle=True, num_workers=2, drop_last=True))
        # training
        for t in range(max_iters):

            # train generator and update gen_x and gen_y histories
            self._train_generator(self._gen, self._disc_x, self._disc_y, gen_steps_per_iter)
            # train x discriminator
            self._train_discriminator(self._disc_x, x_disc_steps_per_iter, batch_size)
            # train y discriminator
            self._train_discriminator(self._disc_y, y_disc_steps_per_iter, batch_size)

            # evaluating and saving models
            if t % eval_interval == 0:
                # evaluate and save generator.
                self._eval_generator(self._gen, self._disc_x, self._disc_y)
                # evaluate and save x discriminator.
                self._eval_discriminator(self._disc_x)  # todo - fix arguments after implementation
                # evaluate and save y discriminator.
                self._eval_discriminator(self._disc_y)  # todo - fix arguments after implementation

    def _train_generator(self, generator, x_discriminator, y_discriminator, num_steps):  # todo - finish
        """"""
        for k in range(num_steps):
            # get z batch
            z_batch = self._z_data_loader.next()

    def _train_discriminator(self, discriminator, num_steps, batch_size):  # todo - finish
        """"""
        for k in range(num_steps):
            # get full x batch (real + generated)
            full_x_batch = self._get_x_batch(batch_size)

    def _eval_generator(self, generator, x_discriminator, y_discriminator):  # todo - implement
        """"""
        pass

    def _eval_discriminator(self, discriminator):  # todo - implement
        """"""
        pass

    def _get_x_batch(self, batch_size):
        """
        :param batch_size: size of the full batch.
        :return: a batch of real and generated x data.
        """
        return self._get_full_batch(self._x_data_loader, self._gen_x_history_buffer, batch_size)

    def _get_y_batch(self, batch_size):
        """
        :param batch_size: size of the full batch.
        :return: a batch of real and generated x data.
        """
        return self._get_full_batch(self._y_data_loader, self._gen_y_history_buffer, batch_size)

    def _get_full_batch(self, data_loader, history_buffer, batch_size):
        """
        :param data_loader: data loader for real dataset.
        :param history_buffer: history buffer for generated data set.
        :param batch_size: size of the full batch.
        :return: a batch of real and generated x data.
        """
        # get real batch
        spects, labels = data_loader.next()
        # calculate number of samples needed to fill the batch size.
        gap_size = batch_size - labels.shape[0]
        assert gap_size > 0  # todo - add message
        # get generated batch
        gen_spects, gen_labels = self._get_batch_from_history(history_buffer, gap_size)
        # return concatenated full batch
        return _concat_batches((spects, gen_spects)), _concat_batches((labels, gen_labels))

    @staticmethod
    def _get_batch_from_history(history, batch_size, label=FALSE_LABEL):
        """
        :param history: history container
        :param batch_size: number of samples to retrieve.
        :param label: (optional) label for the batch (currently all samples from
                      history should have the same label).
        :return: a batch from history
        """
        inds = np.random.randint(history.shape[0], size=batch_size)
        return history[inds], torch.full(size=(inds, 1), fill_value=label)


def _concat_batches(batches):
    """
    Concatenates an iterable of batches to a single batch. Concatenation is done along the first dimension.
    :param batches: iterable of batches.
    """
    return torch.cat(batches, dim=0)


class InfiniteDataLoader(object):
    """
    Wrapper for DataLoader, enables infinite data loading.
    """
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._dataloader_iterator = iter(dataloader)

    def next(self):
        try:
            data = next(self._dataloader_iterator)
        except StopIteration:
            self._dataloader_iterator = iter(self._dataloader)
            data = next(self._dataloader_iterator)
        yield data
