class CustomError(Exception):
    """Exception raised for to stop the forward pass.
    """

    def __init__(self):
        super().__init__()