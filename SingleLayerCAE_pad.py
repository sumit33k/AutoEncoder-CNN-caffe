    def _pad(self, input_x, filter_side):
        """
        pads input_x with the right amount of zeros.
        Args:
            input_x: 4-D tensor, [batch_side, widht, height, depth]
            filter_side: used to dynamically determine the padding amount
        Returns:
            input_x padded
        """
        # calculate the padding amount for each side
        amount = filter_side - 1
        # pad the input on top, bottom, left, right, with amount zeros
        return tf.pad(input_x,
                      [[0, 0], [amount, amount], [amount, amount], [0, 0]])