import abc


class BaseDBCreator(abc.ABC):
    """
    Base class for database creators
    """
    def __init__(self):
        """
        Base class for database creators
        """
        super().__init__()

    @abc.abstractmethod
    def connect(self):
        """
        Connects to the database
        """
        pass

    @abc.abstractmethod
    def write(self, *args, **kwargs):
        """
        Writes the data
        """
        pass

    @abc.abstractmethod
    def close(self):
        """
        Closes the database connection
        """
        pass

    @abc.abstractmethod
    def clean(self):
        """
        Cleans the database
        """
        pass

