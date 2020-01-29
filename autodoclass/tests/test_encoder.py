import pytest
import gensim
from autodoclass import encoder
from autodoclass import reader
from autodoclass import tokenizer

class TestEncoder:
    def test_init(self):
        """ Test the initialization of the Encoder class

        :return None:
        """
        assert encoder.Encoder()

    def test_init_kwargs(self):
        """ Test the initialization of the Encoder class with kwargs

        :return None:
        """
        kwargs = {'argument':None}
        assert encoder.Encoder(**kwargs)

    def test_train_exception(self, dummy_folder):
        """ Test training on encoder
        Should raise exception on Encoder class

        :return None:
        """
        with pytest.raises(NotImplementedError):
            encoder.Encoder().train(dummy_folder)

    def test_encode_exception(self, dummy_text):
        """ Test text encode on encoder
        Should raise exception on Encoder class

        :return None:
        """
        with pytest.raises(NotImplementedError):
            encoder.Encoder().encode(dummy_text)

    def test_serialize(self):
        """ Test serilization of the model

        :return None:
        """
        assert encoder.Encoder().serialize()

    def test_deserialize(self):
        """ Test deserilization of the model

        :return None:
        """
        serialized = encoder.Encoder().serialize()
        assert encoder.Encoder().deserialize(serialized)


class TestEncodedInputIterator:
    def test_init(self, dummy_folder):
        """ Test initialization for the class

        :return None:
        """
        assert encoder.EncodedInputIterator(
            reader.DocumentReader(dummy_folder),
            tokenizer.WhitespaceTokenizer(),
            "Prefix_{}"
        )

    def test_iter(self, dummy_folder):
        """ Test iterating outputs

        :return None:
        """
        iterator = encoder.EncodedInputIterator(
            reader.DocumentReader(dummy_folder),
            tokenizer.WhitespaceTokenizer(),
            "Prefix_{}"
        )
        for train_input in iterator:
            assert isinstance(
                train_input,
                gensim.models.doc2vec.TaggedDocument
            )


class TestDocumentEncoder:
    def test_init(self):
        """ Test initialization of the class

        :return None:
        """
        assert encoder.DocumentEncoder()

    def test_train(self, dummy_folder):
        """ Test training the encoder

        :return None:
        """
        assert encoder.DocumentEncoder().train(dummy_folder)
