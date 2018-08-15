import os
import socket
import getpass
import numpy as np
from datetime import datetime


def make_dir(d):
    """Make directory if it does not exist."""
    if not os.path.exists(d):
        os.makedirs(d)


class data:
    """Class for packaging model results for the circuit nips 2018 paper.

    Params::

    validation_accuracy
        List of model validation accuracies. Each entry is a step of
        validation. We are evaluating this every 500 steps of training.
        Validation must be the average performance across 1000 randomly
        selected exemplars.
    train_accuracy
        List of model training accuracies. Each entry is a step of training.
    model_name
        String with an arbitrary model name.
    """
    def __init__(
            self,
            train_batch_size,
            test_batch_size,
            test_iters,
            shuffle_test,
            shuffle_train,
            lr,
            training_routine,
            loss_function,
            optimizer,
            model_name,
            train_dataset,
            test_dataset,
            output_directory,
            summary_dir,
            checkpoint_dir,
            prediction_directory,
            parameter_count,
            exp_label):
        """Set model information as attributes."""
        self.create_meta(exp_label)
        self.test_loss = []
        self.test_accuracy = []
        self.test_arand = []
        self.test_step = []
        self.test_lr = []
        self.test_pr = []
        self.test_lr_info = {}
        self.train_loss = []
        self.train_accuracy = []
        self.train_arand = []
        self.train_pr = []
        self.train_step = []
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.test_iters = test_iters
        self.shuffle_test = shuffle_test
        self.shuffle_train = shuffle_train
        self.lr = lr
        self.training_routine = training_routine
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.output_directory = output_directory
        self.summary_dir = summary_dir
        self.checkpoint_dir = checkpoint_dir
        self.prediction_directory = prediction_directory
        self.parameter_count = parameter_count
        self.error_message = None
        if output_directory is not None:
            self.file_pointer = os.path.join(
                output_directory,
                exp_label)
        else:
            self.file_pointer = exp_label
        make_dir(self.output_directory)
        self.validate()

    def required_keys(self):
        """Keys we need from data."""
        return [
            # 'test_accuracy',
            # 'train_accuracy',
            # 'test_arand',
            # 'train_arand',
            'directory_pointer',
            'username',
            'homedir',
            'hostname',
            'exp_label'
        ]

    def create_meta(self, exp_label=None):
        """Create meta information about this model."""
        # Get username
        self.username = getpass.getuser()
        self.homedir = os.environ['HOME']
        self.hostname = socket.gethostname()
        if exp_label is None:
            self.exp_label = str(
                datetime.now()).replace(
                ' ', '_').replace(
                ':', '_').replace('-', '_')
        else:
            self.exp_label = exp_label

    def validate(self):
        """Validate that all information is included."""
        keys = self.required_keys()
        assert [k in keys for k in self.__dict__.keys()]
        assert isinstance(self.test_loss, list),\
            'Pass a list of test losses'
        assert isinstance(self.train_loss, list),\
            'Pass a list of training losses'

    def update_training(self, **kwargs):
        """Update training performance."""
        for k, v in kwargs.iteritems():
            data = getattr(self, k)
            if 'info' in k:
                data = v
            else:
                data += [v]
            setattr(self, k, data)

    def update_test(self, **kwargs):
        """Update test performance."""
        for k, v in kwargs.iteritems():
            data = getattr(self, k)
            if 'info' in k:
                data = v
            else:
                data += [v]
            setattr(self, k, data)

    def update_error(self, msg):
        """Pass an error message."""
        self.error_message = msg

    def save(self):
        """Save a npz with model info."""
        np.savez(
            self.file_pointer,
            **self.__dict__)
