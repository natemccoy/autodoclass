import pytest
import gensim
from autodoclass import reader
from autodoclass import tokenizer
from autodoclass import iterator


class TestEncodedInputIterator:
    def test_init(self, dummy_folder):
        """ Test initialization for the class

        :return None:
        """
        assert iterator.EncodedInputIterator(
            reader.DocumentReader(dummy_folder),
            tokenizer.WhitespaceTokenizer(),
            "Prefix_{}"
        )

    def test_iter(self, dummy_folder):
        """ Test iterating outputs

        :return None:
        """
        iter = iterator.EncodedInputIterator(
            reader.DocumentReader(dummy_folder),
            tokenizer.WhitespaceTokenizer(),
            "Prefix_{}"
        )
        for train_input in iter:
            assert isinstance(
                train_input,
                gensim.models.doc2vec.TaggedDocument
            )


class TestDocumentEncodedInputIterator:
    def test_init(self, dummy_folder):
        """ Test initialization for the class

        :return None:
        """
        assert iterator.DocumentEncodedInputIterator(dummy_folder)

    def test_iter(self, dummy_folder):
        """ Test iterating outputs

        :return None:
        """
        for train_input in iterator.DocumentEncodedInputIterator(dummy_folder):
            assert isinstance(
                train_input,
                gensim.models.doc2vec.TaggedDocument
            )


class TestLineEncodedInputIterator:
    def test_init(self, dummy_folder):
        """ Test initialization for the class

        :return None:
        """
        assert iterator.LineEncodedInputIterator(dummy_folder)

    def test_iter(self, dummy_folder):
        """ Test iterating outputs

        :return None:
        """
        for train_input in iterator.LineEncodedInputIterator(dummy_folder):
            assert isinstance(
                train_input,
                gensim.models.doc2vec.TaggedDocument
            )
