class CustomError(Exception):
    """Exception raised to stop the forward pass.
    """

    def __init__(self):
        super().__init__()
