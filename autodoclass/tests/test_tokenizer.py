import pytest
from autodoclass import tokenizer


class TestTokenizer:
    def test_init(self):
        """ Test the initialization of the Tokenizer class

        :return None:
        """
        tokenizer.Tokenizer()

    def test_transform(self, dummy_text):
        """ Test transforming an input text. Should fail in Tokenizer class

        :return None:
        """
        with pytest.raises(NotImplementedError):
            tokenizer.Tokenizer().transform(dummy_text)
