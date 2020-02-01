import pytest
from autodoclass import autodoclass


class TestAutodoclass:
    def test_init(self):
        """ Test initialization of the class

        :return None:
        """
        assert autodoclass.Autodoclass()
