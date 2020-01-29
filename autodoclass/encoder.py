import gensim
import tempfile
from autodoclass import reader
from autodoclass import tokenizer

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

    def encode(self, text):
        """ Encode the text
        Raise exception on base Encoder Model

        :return numpy.array:
        """
        raise NotImplementedError("encode not implemented")

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


class EncodedInputIterator:
    def __init__(self, text_reader, tokenizer, label_format):
        """ Initialize the encoded input iterator

        :param str label_format: prefix format for labels on LabeledSentence
        :param autodoclass.tokenizer.Tokenizer:
        :param autodoclass.reader.Reader:
        """
        self.text_reader = text_reader
        self.tokenizer = tokenizer
        self.label_format = label_format

    def __iter__(self):
        """ Iterate inputs to format consistent with model inputs

        :yield gensim.models.doc2vec.LabeledSentence:
        """
        for text_index, text in enumerate(self.text_reader.yield_texts()):
            yield gensim.models.doc2vec.TaggedDocument(
                words = self.tokenizer.transform(text),
                tags = [self.label_format.format(text_index)]
            )


class DocumentEncoder(Encoder):
    def train(self, folder):
        """ Train the document encoder from the folder

        :return self:
        """
        iterator = EncodedInputIterator(
            reader.DocumentReader(folder),
            tokenizer.WhitespaceTokenizer(),
            "Document_{}"
        )

        self.model.build_vocab(iterator)
        self.model.train(iterator, total_words = self.model.corpus_count)

        return self
