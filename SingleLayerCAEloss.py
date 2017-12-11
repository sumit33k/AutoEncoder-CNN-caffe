    def loss(self, predictions, real_values):
        """Return the loss operation between predictions and real_values.
        Add L2 weight decay term if any.

        Args:
            predictions: predicted values
            real_values: real values

        Returns:
            Loss tensor of type float.
        """
        with tf.variable_scope('loss'):
            # 1/2n \sum^{n}_{i=i}{(x_i - x'_i)^2}
            mse = tf.div(tf.reduce_mean(
                tf.square(tf.subtract(predictions, real_values))),
                         2,
                         name="mse")
            tf.add_to_collection('losses', mse)
            
            # mse + weight_decay per layer
            error = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return error