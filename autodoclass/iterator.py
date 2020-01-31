import gensim
from autodoclass import reader
from autodoclass import tokenizer
from autodoclass import configuration


class EncodedInputIterator:
    """ Iterate TaggedDocument objects needed for encoders to encode text inputs.
    """
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


class DocumentEncodedInputIterator(EncodedInputIterator):
    """ Iterate TaggedDocument objects needed for encoders to encode
    text inputs on the Document Level
    """
    def __init__(self, folder):
        """ Initialize the iterator for Document Encoding Inputs

        :param str folder: folder to read text input documents
        """
        super().__init__(
            reader.DocumentReader(folder),
            configuration.DEFAULT_TOKENIZER,
            "Document_{}"
        )


class LineEncodedInputIterator(EncodedInputIterator):
    """ Iterate TaggedDocument objects needed for encoders to encode
    text inputs on the Document Level
    """
    def __init__(self, folder):
        """ Initialize the iterator for Line Encoding Inputs

        :param str folder: folder to read text input documents
        """
        super().__init__(
            reader.LineReader(folder),
            configuration.DEFAULT_TOKENIZER,
            "Line_{}"
        )
