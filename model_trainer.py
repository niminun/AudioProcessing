import argparse
import os
import time
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from data_processing import SpectrogramDataset
from models import ResNetDiscriminator, ResNetGenerator, NonPaddedBasicBlock, LeakyBasicBlock

MAG_DB_BIN = "__mag_db.bin"
DISCRIMINATOR_Y = "discriminator_y"
DISCRIMINATOR_X = "discriminator_x"
GENERATOR = "generator"

SAMPLE_W = 64
SAMPLE_H = 1025
HIST_LEN = 256

TRUE_LABEL = 1
FALSE_LABEL = 0
THRESH = 0.5

LR = 1e-3
LR_GAMMA = 0.1
STEPS = 4
WEIGHT_DECAY = 5e-5

N_WORKERS = 2
# CUDA device check
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class VoiceSeparationTrainer(object):
    """
    A class for the purpose of training all required models for voice separation.
    """

    def __init__(self, base_dir, generator, discriminator_x, discriminator_y,
                 sample_height=SAMPLE_H, sample_width=SAMPLE_W, history_len=HIST_LEN):
        """
        :param base_dir: base dir for all outputs.
        :param generator: a model that generates Y distributed data given Z=(X+Y) distributed data.
        :param discriminator_x: a model that discriminates between generated X data and real X data.
        :param discriminator_y: a model that discriminates between generated Y data and real Y data.
        :param sample_height: (optional) height of input sample.
        :param sample_width: (optional) width of input sample.
        :param history_len: (optional) size of history buffer.
        """
        self._base_dir = base_dir
        self._gen = generator
        self._disc_x = discriminator_x
        self._disc_y = discriminator_y
        self._sample_height = sample_height
        self._sample_width = sample_width
        self._history_len = history_len
        self._train_gen_x_history = \
            torch.zeros((self._history_len, 1, self._sample_height, self._sample_width)).to(device)
        self._train_gen_y_history = \
            torch.zeros((self._history_len, 1, self._sample_height, self._sample_width)).to(device)
        self._val_gen_x_history = \
            torch.zeros((self._history_len, 1, self._sample_height, self._sample_width)).to(device)
        self._val_gen_y_history = \
            torch.zeros((self._history_len, 1, self._sample_height, self._sample_width)).to(device)
        self._train_x_data_loader = None
        self._train_y_data_loader = None
        self._train_z_data_loader = None
        self._val_x_data_loader = None
        self._val_y_data_loader = None
        self._val_z_data_loader = None

    def train(self, train_x_data_dir, train_y_data_dir, train_z_data_dir,
              val_x_data_dir, val_y_data_dir, val_z_data_dir,
              max_iters=1000, gen_steps_per_iter=10, x_disc_steps_per_iter=10,
              y_disc_steps_per_iter=10, eval_interval=None, batch_size=32):
        """
        Training the models
        :param train_x_data_dir: A directory containing X data for training.
        :param train_y_data_dir: A directory containing Y data for training.
        :param train_z_data_dir: A directory containing Z data for training.
        :param val_x_data_dir: A directory containing X data for validation.
        :param val_y_data_dir: A directory containing Y data for validation.
        :param val_z_data_dir: A directory containing Z data for validation.
        :param max_iters: (optional) number of iterations to run.
        :param gen_steps_per_iter: (optional) number of generator training steps to run in each iteration.
        :param x_disc_steps_per_iter: (optional) number of x discriminator training steps to run in each iteration.
        :param y_disc_steps_per_iter: (optional) number of y discriminator training steps to run in each iteration.
        :param eval_interval: (optional) number of iterations between each models evaluation.
        :param batch_size: size of batches to be used for each training iteration of all models.
        """

        eval_interval = eval_interval if eval_interval else int(max_iters / 100)  # do 100 evaluations by default.
        log_file = os.path.join(self._base_dir, "generator_train.log")
        to_log(log_file, "Start Training")
        to_log(log_file, "device: {}".format(device))

        # creating data generators
        train_x_data_set = SpectrogramDataset(in_dir=train_x_data_dir, suffix=MAG_DB_BIN,
                                              sample_width=self._sample_width,
                                              spect_h=self._sample_height, label=TRUE_LABEL)
        train_y_data_set = SpectrogramDataset(in_dir=train_y_data_dir, suffix=MAG_DB_BIN,
                                              sample_width=self._sample_width,
                                              spect_h=self._sample_height, label=TRUE_LABEL)
        train_z_data_set = SpectrogramDataset(in_dir=train_z_data_dir, suffix=MAG_DB_BIN,
                                              sample_width=self._sample_width,
                                              spect_h=self._sample_height, label=TRUE_LABEL)
        val_x_data_set = SpectrogramDataset(in_dir=val_x_data_dir, suffix=MAG_DB_BIN,
                                            sample_width=self._sample_width,
                                            spect_h=self._sample_height, label=TRUE_LABEL)
        val_y_data_set = SpectrogramDataset(in_dir=val_y_data_dir, suffix=MAG_DB_BIN,
                                            sample_width=self._sample_width,
                                            spect_h=self._sample_height, label=TRUE_LABEL)
        val_z_data_set = SpectrogramDataset(in_dir=val_z_data_dir, suffix=MAG_DB_BIN,
                                            sample_width=self._sample_width,
                                            spect_h=self._sample_height, label=TRUE_LABEL)
        half_bs = int(batch_size/2)
        self._train_x_data_loader = InfiniteDataLoader(DataLoader(dataset=train_x_data_set, batch_size=half_bs,
                                                                  shuffle=True, num_workers=N_WORKERS, drop_last=True))
        self._train_y_data_loader = InfiniteDataLoader(DataLoader(dataset=train_y_data_set, batch_size=half_bs,
                                                                  shuffle=True, num_workers=N_WORKERS, drop_last=True))
        self._train_z_data_loader = InfiniteDataLoader(DataLoader(dataset=train_z_data_set, batch_size=half_bs,
                                                                  shuffle=True, num_workers=N_WORKERS, drop_last=True))
        self._val_x_data_loader = InfiniteDataLoader(DataLoader(dataset=val_x_data_set, batch_size=half_bs,
                                                                shuffle=True, num_workers=N_WORKERS, drop_last=True))
        self._val_y_data_loader = InfiniteDataLoader(DataLoader(dataset=val_y_data_set, batch_size=half_bs,
                                                                shuffle=True, num_workers=N_WORKERS, drop_last=True))
        self._val_z_data_loader = InfiniteDataLoader(DataLoader(dataset=val_z_data_set, batch_size=half_bs,
                                                                shuffle=True, num_workers=N_WORKERS, drop_last=True))
        # training
        gen_optimizer = torch.optim.Adam(self._gen.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        disc_x_optimizer = torch.optim.Adam(self._disc_x.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        disc_y_optimizer = torch.optim.Adam(self._disc_y.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        milestones = [int((i / STEPS) * max_iters) for i in range(1, STEPS)]
        gen_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(gen_optimizer, milestones=milestones,
                                                                gamma=LR_GAMMA)
        disc_x_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(disc_x_optimizer, milestones=milestones,
                                                                   gamma=LR_GAMMA)
        disc_y_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(disc_y_optimizer, milestones=milestones,
                                                                   gamma=LR_GAMMA)
        best_gen_loss = np.inf
        best_disc_x_loss = np.inf
        best_disc_y_loss = np.inf
        for t in range(max_iters):
            # train generator and update gen_x and gen_y histories
            self._train_generator(self._gen, self._disc_x, self._disc_y, gen_optimizer, gen_steps_per_iter)
            gen_lr_scheduler.step()
            # train x discriminator
            self._train_discriminator(self._disc_x, disc_x_optimizer, self._get_x_batch,
                                      x_disc_steps_per_iter, batch_size, name=DISCRIMINATOR_X)
            disc_x_lr_scheduler.step()
            # train y discriminator
            self._train_discriminator(self._disc_y, disc_y_optimizer, self._get_y_batch,
                                      y_disc_steps_per_iter, batch_size, name=DISCRIMINATOR_Y)
            disc_y_lr_scheduler.step()

            # evaluating and saving models
            if t % eval_interval == 0:
                to_log(log_file, '-' * 36 + ' Validation - {:d} '.format(t / eval_interval) + '-' * 36)
                # evaluate and save generator.
                gen_loss, gen_x_success_rate, gen_y_success_rate = \
                    self._validate_generator(self._gen, self._disc_x, self._disc_y, gen_steps_per_iter * 100)
                if gen_loss <= best_gen_loss:
                    self._save(self._gen, os.path.join(self._base_dir, GENERATOR))
                # evaluate and save x discriminator.
                disc_x_loss, disc_x_precision, disc_x_recall = \
                    self._validate_discriminator(self._disc_x, self._get_x_batch, x_disc_steps_per_iter * 100,
                                                 batch_size, name=DISCRIMINATOR_X)
                if disc_x_loss <= best_disc_x_loss:
                    self._save(self._disc_x, os.path.join(self._base_dir, DISCRIMINATOR_X))
                # evaluate and save y discriminator.
                disc_y_loss, disc_y_precision, disc_y_recall = \
                    self._validate_discriminator(self._disc_y, self._get_y_batch, y_disc_steps_per_iter * 100,
                                                 batch_size, name=DISCRIMINATOR_Y)
                if disc_y_loss <= best_disc_y_loss:
                    self._save(self._disc_y, os.path.join(self._base_dir, DISCRIMINATOR_Y))
                to_log(log_file, '-' * 89)

    def _train_generator(self, generator, x_discriminator, y_discriminator, optimizer, num_steps):
        """"""
        log_file = os.path.join(self._base_dir, "generator_train.log")
        generator.train()  # turn on training mode - enable train-only stages such as dropout batchNorm, etc'
        for k in range(num_steps):
            start_time = time.time()
            # get z batch
            z_batch = self._train_z_data_loader.next()
            # generate x data, and subtract from z to get y data
            x_generated = generator(z_batch)
            y_generated = z_batch - x_generated  # todo - test this
            # get discriminators' predictions, calculate loss and update generator
            x_disc_pred = x_discriminator(x_generated)
            y_disc_pred = y_discriminator(y_generated)
            x_loss = self._calc_gen_loss(x_disc_pred)
            y_loss = self._calc_gen_loss(y_disc_pred)
            generator.zero_grad()
            x_loss.backward()
            y_loss.backward()
            optimizer.step()
            # updating generated data history.
            self._update_history(x_generated, self._train_gen_x_history)
            self._update_history(y_generated, self._train_gen_y_history)
            step_time = time.time() - start_time
            to_log(log_file, '| {:s} |  step {:3d}  |  ms/batch {:5.2f}  |  x_loss {:5.3f}  |  y_loss {:5.3f}  |'
                             .format(time.strftime("%X"), k, step_time * 1000, x_loss, y_loss))

    def _train_discriminator(self, discriminator, optimizer, batch_load_fn, num_steps, batch_size, name):
        """"""
        log_file = os.path.join(self._base_dir, "{}_train.log".format(name))
        discriminator.train()  # turn on training mode - enable train-only stages such as dropout batchNorm, etc'
        for k in range(num_steps):
            start_time = time.time()
            # get full x batch (real + generated)
            inputs, targets = batch_load_fn(batch_size, is_train=True)
            inputs, targets = inputs.to(device), targets.to(device)
            # predict, calculate loss and update weights
            pred = discriminator(inputs)
            loss = self._calc_disc_loss(pred, targets)
            discriminator.zero_grad()
            loss.backward()
            optimizer.step()
            step_time = time.time() - start_time
            to_log(log_file, '| {:s} |  step {:3d}  |  ms/batch {:5.2f}  |  {:s}_loss {:5.3f}  |'
                   .format(time.strftime("%X"), k, step_time * 1000, name, loss))

    def _validate_generator(self, generator, x_discriminator, y_discriminator, num_steps):
        """"""
        log_file = os.path.join(self._base_dir, "generator_train.log")
        total_loss, x_success, y_success = 0, 0, 0
        y_success = 0
        # turn on eval mode - disable train-only stages such as dropout batchNorm, etc'
        generator.eval()
        x_discriminator.eval()
        y_discriminator.eval()
        # evaluate model
        with torch.no_grad():
            start_time = time.time()
            for k in range(num_steps):
                # get z batch
                z_batch = self._val_z_data_loader.next()
                # generate x data, and subtract from z to get y data
                x_generated = generator(z_batch)
                y_generated = z_batch - x_generated  # todo - test this
                # get discriminators' predictions, calculate loss and stats and add to the total loss.
                x_disc_pred = x_discriminator(x_generated)
                y_disc_pred = y_discriminator(y_generated)
                x_success += torch.sum(x_disc_pred >= THRESH)
                y_success += torch.sum(y_disc_pred >= THRESH)
                total_loss += self._calc_gen_loss(x_disc_pred)
                total_loss += self._calc_gen_loss(y_disc_pred)
                # updating generated data history.
                self._update_history(x_generated, self._val_gen_x_history)
                self._update_history(y_generated, self._val_gen_y_history)
        val_time = time.time() - start_time
        # summarize losses
        avg_loss = float(total_loss) / num_steps
        x_success_rate = float(x_success) / (num_steps * self._val_z_data_loader.batch_size)
        y_success_rate = float(y_success) / (num_steps * self._val_z_data_loader.batch_size)
        to_log(log_file, '| {:s} |  validation time {:5.2f} ms  |  validation loss {:5.3f}  '
                         '|  x success rate {:5.3f}  |  y_success rate {:5.3f}  |'
                         .format(time.strftime("%X"), val_time*1000, avg_loss, x_success_rate, y_success_rate))
        return avg_loss, x_success_rate, y_success_rate

    def _validate_discriminator(self, discriminator, batch_load_fn, num_steps, batch_size, name):
        """"""
        log_file = os.path.join(self._base_dir, "{}_train.log".format(name))
        total_loss = 0
        pos_count = 0  # positive labels counter
        tp_count = 0  # true positive counter
        fp_count = 0  # false positive counter
        # turn on eval mode - disable train-only stages such as dropout batchNorm, etc'
        discriminator.eval()
        # evaluate model
        with torch.no_grad():
            start_time = time.time()
            for k in range(num_steps):
                # get full x batch (real + generated)
                inputs, targets = batch_load_fn(batch_size, is_train=False)
                inputs, targets = inputs.to(device), targets.to(device)
                # predict, calculate loss and update total loss.
                pred = discriminator(inputs)
                pos_targets = (targets == TRUE_LABEL)
                neg_targets = (targets == FALSE_LABEL)
                pos_pred = (pred >= THRESH)
                assert torch.sum(pos_targets) + torch.sum(neg_targets) == targets.shape[0]
                pos_count += torch.sum(pos_targets)
                tp_count += torch.sum(pos_targets & pos_pred)
                fp_count += torch.sum(neg_targets & pos_pred)
                total_loss += self._calc_disc_loss(pred, targets)
        # summarize losses
        avg_loss = float(total_loss) / num_steps
        precision = float(tp_count) / pos_count
        recall = float(tp_count) / (tp_count + fp_count)
        val_time = time.time() - start_time
        to_log(log_file, '| {:s} |  validation time {:5.2f} ms  |  validation loss {:5.3f}  |  precision {:5.3f}  |'
                         '  recall {:5.3f}  |'.format(time.strftime("%X"), val_time*1000, avg_loss, precision, recall))
        return avg_loss, precision, recall

    def _get_x_batch(self, batch_size, is_train):
        """
        :param batch_size: size of the full batch.
        :param is_train: is it a train batch (otherwise a validation batch).
        :return: a batch of real and generated x data.
        """
        if is_train:
            return self._get_full_batch(self._train_x_data_loader, self._train_gen_x_history, batch_size)
        else:
            return self._get_full_batch(self._val_x_data_loader, self._val_gen_x_history, batch_size)

    def _get_y_batch(self, batch_size, is_train):
        """
        :param batch_size: size of the full batch.
        :param is_train: is it a train batch (otherwise a validation batch).
        :return: a batch of real and generated x data.
        """
        if is_train:
            return self._get_full_batch(self._train_y_data_loader, self._train_gen_y_history, batch_size)
        else:
            return self._get_full_batch(self._val_y_data_loader, self._val_gen_y_history, batch_size)

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
        assert gap_size >= 0
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

    @staticmethod
    def _update_history(generated, history, replace_size=None):
        # sampling indices for replacements
        hist_len = history.shape[0]
        gen_len = generated.shape[0]
        if not replace_size:
            replace_size = int(gen_len / 2.0)
        gen_rep_inds = np.random.choice(gen_len, replace_size)
        hist_rep_inds = np.random.choice(hist_len, replace_size)
        # updating history
        history[hist_rep_inds, :, :, :] = generated[gen_rep_inds, :, :, :]

    @staticmethod
    def _calc_disc_loss(pred, targets):
        loss_function = nn.MSELoss()
        return loss_function(pred, targets)

    @staticmethod
    def _calc_gen_loss(disc_pred):
        loss_function = nn.MSELoss()
        return loss_function(disc_pred, torch.ones_like(disc_pred))

    @staticmethod
    def _save(model, save_path):
        with open(save_path, 'wb') as model_path:
            torch.save(model, model_path)

    @staticmethod
    def _load(model_path):
        with open(model_path, 'rb') as model_path:
            return torch.load(model_path, map_location=device)


def _concat_batches(batches):
    """
    Concatenates an iterable of batches to a single batch. Concatenation is done along the first dimension.
    :param batches: iterable of batches.
    """
    return torch.cat(batches, dim=0)


def to_log(log_file, message, mode="a"):
    with open(log_file, mode) as f:
        print(message, file=f)


class InfiniteDataLoader(object):
    """
    Wrapper for DataLoader, enables infinite data loading.
    """
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._dataloader_iterator = iter(dataloader)
        self.batch_size = self._dataloader.batch_size

    def next(self):
        try:
            data = next(self._dataloader_iterator)
        except StopIteration:
            self._dataloader_iterator = iter(self._dataloader)
            data = next(self._dataloader_iterator)
        yield data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Voice separation model trainer.')
    parser.add_argument('--base_dir', required=True, help='a path to model training output dir.')
    parser.add_argument('--x_data_dir', required=True, help='a path to x data dir.')
    parser.add_argument('--y_data_dir', required=True, help='a path to y data dir.')
    parser.add_argument('--z_data_dir', required=True, help='a path to z data dir.')
    args = parser.parse_args()

    _train_x_data = os.path.join(args.x_data_dir, "train")
    _train_y_data = os.path.join(args.y_data_dir, "train")
    _train_z_data = os.path.join(args.z_data_dir, "train")
    _val_x_data = os.path.join(args.x_data_dir, "validation")
    _val_y_data = os.path.join(args.y_data_dir, "validation")
    _val_z_data = os.path.join(args.z_data_dir, "validation")

    _generator = ResNetGenerator(NonPaddedBasicBlock, [2, 1, 1, 1])  # TODO - try sizes
    _discriminator_x = ResNetDiscriminator(LeakyBasicBlock, [2, 2, 2, 2])  # TODO - try sizes
    _discriminator_y = ResNetDiscriminator(LeakyBasicBlock, [2, 2, 2, 2])  # TODO - try sizes
    _trainer = VoiceSeparationTrainer(args.base_dir, _generator, _discriminator_x, _discriminator_y)
    _trainer.train(train_x_data_dir=_train_x_data, train_y_data_dir=_train_y_data, train_z_data_dir=_train_z_data,
                   val_x_data_dir=_val_x_data, val_y_data_dir=_val_y_data, val_z_data_dir=_val_z_data)

