import spacy


class Tokenizer:
    def transform(self, text):
        """ Transform a given text input into tokens.
        In tokenizer class, raise exception

        :param str text: input text to tokenize
        :raises NotImplementedError:
        """
        raise NotImplementedError("transform Not Implemented")


class WhitespaceTokenizer(Tokenizer):
    def transform(self, text):
        """ Transform a given text input into tokens

        :param str text: input text to tokenize
        :return list(str): list of tokens (strings)
        """
        return text.split()


class SpacyTokenizer(Tokenizer):
    def __init__(self, language = 'en_core_web_sm'):
        """ Initialize the Tokenizer

        :param str language: spacy language model name
        :return None:
        """
        self.nlp = spacy.load(
            language,
            disable = ['ner', 'parser', 'tagger', 'textcat']
        )

    def transform(self, text):
        """ Transform a given text input into tokens

        :param str text: input text to tokenize
        :return list(str): list of tokens (strings)
        """
        return list(map(str, self.nlp(text)))
