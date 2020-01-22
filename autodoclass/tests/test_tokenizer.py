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


class TestWhitespaceTokenizer:
    def test_init(self):
        """ Test the initialization for the WhitespaceTokenizer class

        :return None:
        """
        tokenizer.WhitespaceTokenizer()

    def test_transform(self, dummy_text):
        """ Test calling the transform function

        :return None:
        """
        assert tokenizer.WhitespaceTokenizer().transform(dummy_text)

    def test_transform_types(self, dummy_text):
        """ Test the output types from the transform function

        :return None:
        """
        tokens = tokenizer.WhitespaceTokenizer().transform(dummy_text)
        assert isinstance(tokens, list)
        assert isinstance(tokens.pop(), str)


class TestSpacyTokenizer:
    def test_init(self):
        """ Test the initialization for the SpacyTokenizer class

        :return None:
        """
        tokenizer.SpacyTokenizer()

    def test_transform(self, dummy_text):
        """ Test calling the transform function

        :return None:
        """
        assert tokenizer.SpacyTokenizer().transform(dummy_text)

    def test_transform_types(self, dummy_text):
        """ Test the output types from the transform function

        :return None:
        """
        tokens = tokenizer.WhitespaceTokenizer().transform(dummy_text)
        assert isinstance(tokens, list)
        assert isinstance(tokens.pop(), str)
