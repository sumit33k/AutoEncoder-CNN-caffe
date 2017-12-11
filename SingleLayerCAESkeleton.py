# import tensorflow
import tensorflow as tf
# import the utils file in the current folder
from . import utils
# from the Autoencoder file import the interface to implement
from .Autoencoder import Autoencoder


class SingleLayerCAE(Autoencoder):
    """ Build a single layer CAE"""

    def get(self, images, train_phase=False, l2_penalty=0.0):
        """ define the model with its inputs.
        Use this function to define the model in training and when exporting the model
        in the protobuf format.

        Args:
            images: model input
            train_phase: set it to True when defining the model, during train
            l2_penalty: float value, weight decay (l2) penalty

        Return:
            is_training_: tf.bool placeholder enable/disable training ops at run time
            predictions: the model output
        """
        pass

    def loss(self, predictions, real_values):
        """Return the loss operation between predictions and real_values.
        Add L2 weight decay term if any.

        Args:
            predictions: predicted values
            real_values: real values

        Returns:
            Loss tensor of type float"""
        pass