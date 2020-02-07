import hdbscan
import pickle
from autodoclass import iterator
from autodoclass import configuration


class Clusterer:
    def __init__(self, encoder, **model_kwargs):
        self.encoder = encoder
        self.model = hdbscan.HDBSCAN(prediction_data = True)
        if model_kwargs:
            self.model = hdbscan.HDBSCAN(**model_kwargs)

    def train(self, folder):
        """ Train the clusterer from a folder of documents

        :raises NotImplementedError:
        """
        raise NotImplementedError("train not implemented")

    def predict(self, text):
        """ Predict the clusters from the text input

        :raises NotImplementedError:
        """
        raise NotImplementedError("predict not implemented")

    def serialize(self):
        """ Serialize the clusterer into bytes

        :return bytes:
        """
        return pickle.dumps(self.model)

    def deserialize(self, serialized):
        """ Deserialize the clusterer

        :return self:
        """
        self.model = pickle.loads(serialized)
        return self


class DocumentClusterer(Clusterer):
    def train(self, folder):
        """ Train the Clusterer from a folder of documents

        :return self:
        """
        encoded_inputs_iterator = map(
            lambda x:self.encoder.encode(x.words),
            iterator.DocumentEncodedInputIterator(folder)
        )
        print("Fitting", self)
        self.model.fit(list(encoded_inputs_iterator))
        return self

    def predict(self, text):
        """ Predict the cluster for the input text

        :return tuple: label and stength
        """
        labels, strengths = hdbscan.approximate_predict(
            self.model, [
                self.encoder.encode(
                    configuration.DEFAULT_TOKENIZER.transform(text)
                )
            ]
        )
        return int(labels[0]), float(strengths[0])


class LineClusterer(Clusterer):
    def train(self, folder):
        """ Train the Clusterer from a folder of documents

        :return self:
        """
        encoded_inputs_iterator = map(
            lambda x:self.encoder.encode(x.words),
            iterator.LineEncodedInputIterator(folder)
        )
        print("Fitting", self)
        self.model.fit(list(encoded_inputs_iterator))
        return self

    def predict(self, text):
        """ Predict the cluster for the input text

        :return tuple: label and stength
        """
        labels, strengths = hdbscan.approximate_predict(
            self.model, [
                self.encoder.encode(
                    configuration.DEFAULT_TOKENIZER.transform(text)
                )
            ]
        )
        return int(labels[0]), float(strengths[0])
