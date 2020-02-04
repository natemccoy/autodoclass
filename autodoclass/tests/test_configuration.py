from autodoclass import configuration
from autodoclass import tokenizer


def test_DEFAULT_TOKENIZER():
    """ Test the DEFAULT_TOKENIZER is available and correct type

    :return None:
    """
    assert isinstance(configuration.DEFAULT_TOKENIZER, tokenizer.Tokenizer)
