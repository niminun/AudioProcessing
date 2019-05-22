from torch.utils.data import DataLoader
from data_processing import SpectrogramDataset

SAMPLE_W = 64
SAMPLE_H = 1025
HIST_LEN = 256


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
        self._gen_x_history_buffer = None  # todo - replace with numpy zeros in right size
        self._gen_y_history_buffer = None  # todo - replace with numpy zeros in right size
        self._x_data_loader = None
        self._y_data_loader = None
        self._z_data_loader = None

    def train(self, x_data_dir, y_data_dir, z_data_dir, max_iters=1000, gen_steps_per_iter=10,
              x_disc_steps_per_iter=10, y_disc_steps_per_iter=10):
        """
        Training the models
        :param x_data_dir: A directory containing X data.
        :param y_data_dir: A directory containing Y data.
        :param z_data_dir: A directory containing Z data.
        :param max_iters: (optional) number of iterations to run.
        :param gen_steps_per_iter: (optional) number of generator training steps to run in each iteration.
        :param x_disc_steps_per_iter: (optional) number of x discriminator training steps to run in each iteration.
        :param y_disc_steps_per_iter: (optional) number of y discriminator training steps to run in each iteration.
        """

        # creating data generators
        x_data_set = SpectrogramDataset(in_dir=x_data_dir, suffix="__mag_db.bin",
                                        sample_width=self._sample_width, spect_h=self._sample_height)
        y_data_set = SpectrogramDataset(in_dir=y_data_dir, suffix="__mag_db.bin",
                                        sample_width=self._sample_width, spect_h=self._sample_height)
        z_data_set = SpectrogramDataset(in_dir=z_data_dir, suffix="__mag_db.bin",
                                        sample_width=self._sample_width, spect_h=self._sample_height)
        self._x_data_loader = InfiniteDataLoader(DataLoader(dataset=x_data_set, batch_size=16,
                                                            shuffle=True, num_workers=2, drop_last=True))
        self._y_data_loader = InfiniteDataLoader(DataLoader(dataset=y_data_set, batch_size=16,
                                                            shuffle=True, num_workers=2, drop_last=True))
        self._z_data_loader = InfiniteDataLoader(DataLoader(dataset=z_data_set, batch_size=16,
                                                            shuffle=True, num_workers=2, drop_last=True))

        # training
        for t in range(max_iters):

            for k in range(gen_steps_per_iter):
                # get z batch
                z_batch = self._z_data_loader.next()
                # train generator and update gen_x and gen_y histories
                self._train_generator(self._gen, self._disc_x, self._disc_y, z_batch)

            for k in range(x_disc_steps_per_iter):
                # get full x batch (true + generated)
                full_x_batch = self._get_x_batch()
                # train x discriminator
                self._train_discriminator(self._disc_x, full_x_batch)

            for k in range(y_disc_steps_per_iter):
                # get full y batch (true + generated)
                full_y_batch = self._get_y_batch()
                # train y discriminator
                self._train_discriminator(self._disc_y, full_y_batch)

    def _get_x_batch(self):
        """
        :return: a batch of real and generated x data.
        """
        # get x batch
        x_batch = self._x_data_loader.next()
        # get gen_x batch
        gen_x_batch = self._get_batch_from_history(self._gen_x_history_buffer)
        # train x discriminator
        return self._concat_batches((x_batch, gen_x_batch))

    def _get_y_batch(self):
        """
        :return: a batch of real and generated x data.
        """
        # get y batch
        y_batch = self._y_data_loader.next()
        # get gen_y batch
        gen_y_batch = self._get_batch_from_history(self._gen_y_history_buffer)
        # train x discriminator
        return self._concat_batches((y_batch, gen_y_batch))

    def _train_generator(self, generator, x_discriminator, y_discriminator, batch):
        """"""
        pass

    def _train_discriminator(self, discriminator, batch):
        """"""
        pass

    def _concat_batches(self, batches):
        """
        Concatenates an iterable of batches to a single batch
        :param batches: iterable of batches.
        :return: batch.
        """
        return []

    def _get_batch_from_history(self, history):
        """
        :param history: history container
        :return: a batch from history
        """
        return []


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

