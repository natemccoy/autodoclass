import pickle
import io
import tqdm
from autodoclass import clusterer
from autodoclass import encoder
from autodoclass import configuration


class Autodoclass:
    """ Automatic Text Document Classification
    """
    def __init__(self):
        self.document_encoder = encoder.DocumentEncoder()
        self.line_encoder = encoder.LineEncoder()
        self.document_clusterer = clusterer.DocumentClusterer(self.document_encoder)
        self.line_clusterer = clusterer.LineClusterer(self.line_encoder)

    def train(self, folder):
        """ Train the class from the text files in the folder

        :return self:
        """
        training_steps = [
            self.document_encoder,
            self.line_encoder,
            self.document_clusterer,
            self.line_clusterer,
        ]

        for step_number, step in enumerate(tqdm.tqdm(training_steps), 1):
            print("[{}/{}] Training {}".format(step_number, len(training_steps), step))
            step.train(folder)

        return self

    def predict(self, document_text):
        """ Predict the labels and confidences for the document and line level

        :return tuple:
        """
        document_predictions = self.document_clusterer.predict(document_text)
        lines_predictions = [
            self.line_clusterer.predict(line.rstrip('\n').rstrip('\r'))
            for line in io.StringIO(document_text).readlines()
        ]
        return document_predictions, lines_predictions


    def save(self, model_filename):
        """ Save the model to file

        :return self:
        """
        serialized_model = (
            self.document_encoder.serialize(),
            self.line_encoder.serialize(),
            self.document_clusterer.serialize(),
            self.line_clusterer.serialize()
        )

        with open(model_filename, 'wb') as mf:
            pickle.dump(serialized_model, mf)

        return self

    @classmethod
    def load(cls, model_filename):
        """ Load the model from the filename

        :return Autodoclass:
        """
        with open(model_filename, 'rb') as mf:
            denc_s, lenc_s, dclus_s, lclus_s = pickle.load(mf)

        adc = cls()

        adc.document_encoder.deserialize(denc_s)
        adc.line_encoder.deserialize(lenc_s),
        adc.document_clusterer.deserialize(dclus_s),
        adc.line_clusterer.deserialize(lclus_s)

        return adc
