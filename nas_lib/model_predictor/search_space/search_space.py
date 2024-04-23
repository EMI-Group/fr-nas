


class DatabaseSearchSpace:
    """
    Interface for search space with database
    """
    def __init__(self):
        pass

    def query(self, archs):
        raise NotImplementedError


    def sample_data(self, sample_size):
        raise NotImplementedError