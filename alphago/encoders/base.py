class BaseEncoder:
    def name(self):
        raise NotImplementedError()

    def encode(self, game_state):
        """
        Convert a Go board state into a numeric data structure
        """
        raise NotImplementedError()

    def encode_point(self, point):
        """
        Turn a Go board point into an integer index
        """
        raise NotImplementedError()

    def decode_point_index(self, index):
        """
        Turn an integer index into a Go board point
        """
        raise NotImplementedError()

    def num_points(self):
        """
        Return the number of points on the board
        """
        raise NotImplementedError()

    def shape(self):
        """
        Return the 2D shape of the encoded board state
        """
        raise NotImplementedError()
