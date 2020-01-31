import gensim
import tempfile
from autodoclass import iterator


class Encoder:
    def __init__(self, **kwargs):
        """ Initialize the encoder with Doc2Vec model

        :param dict kwargs: keyword arguments passed into Doc2Vec model
        :return None:
        """
        self.model = gensim.models.doc2vec.Doc2Vec(**kwargs)

    def train(self, folder):
        """ Train the encoder from a given folder
        Raise exception on base Encoder Model

        :return self:
        """
        raise NotImplementedError("train not implemented")

    def encode(self, tokens):
        """ Encode the input text

        :return self:
        """
        return self.model.infer_vector(tokens)

    def serialize(self):
        """ Return serialized form of the encoder

        :return binary:
        """
        with tempfile.NamedTemporaryFile(suffix = ".gz", delete = True) as tf:
            self.model.save(tf.name)
            with open(tf.name, 'rb') as model_f:
                serialized = model_f.read()
        return serialized

    def deserialize(self, serialized):
        """ Load model from serialized form of the encoder

        :return self:
        """
        with tempfile.NamedTemporaryFile(suffix = ".gz", delete = True) as tf:
            with open(tf.name, 'wb') as model_f:
                model_f.write(serialized)
            self.model = gensim.models.doc2vec.Doc2Vec.load(tf.name)
        return self


class DocumentEncoder(Encoder):
    def train(self, folder):
        """ Train the document encoder from the folder

        :return self:
        """
        self.model.build_vocab(
            iterator.DocumentEncodedInputIterator(folder)
        )
        self.model.train(
            iterator.DocumentEncodedInputIterator(folder),
            total_words = self.model.corpus_count,
            epochs = self.model.epochs
        )

        return self


class LineEncoder(Encoder):
    def train(self, folder):
        """ Train the document encoder from the folder

        :return self:
        """
        self.model.build_vocab(
            iterator.LineEncodedInputIterator(folder)
        )
        self.model.train(
            iterator.LineEncodedInputIterator(folder),
            total_words = self.model.corpus_count,
            epochs = self.model.epochs
        )

        return self
