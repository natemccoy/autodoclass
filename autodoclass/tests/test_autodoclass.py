import tempfile
from autodoclass import autodoclass


class TestAutodoclass:
    def test_init(self):
        """ Test initialization of the class

        :return None:
        """
        assert autodoclass.Autodoclass()

    def test_train(self, dummy_folder):
        """ Train the model

        :return None:
        """
        assert autodoclass.Autodoclass().train(dummy_folder)

    def test_save(self, dummy_folder):
        """ Test saving the model

        :return None:
        """
        with tempfile.NamedTemporaryFile(suffix = ".pkl", delete = True) as tf:
            assert autodoclass.Autodoclass().train(dummy_folder).save(tf.name)

    def test_load(self, dummy_folder):
        """ Test loading the file

        :return None:
        """
        with tempfile.NamedTemporaryFile(suffix = ".pkl", delete = True) as tf:
            model = autodoclass.Autodoclass().train(dummy_folder).save(tf.name)
            del model
            assert autodoclass.Autodoclass.load(tf.name)

    def test_predict(self, dummy_folder, dummy_text):
        """ Test predicting with the model

        :return None:
        """
        assert autodoclass.Autodoclass().train(dummy_folder).predict(dummy_text)

    def test_predict_types(self, dummy_folder, dummy_text):
        """ Test predicting with the model

        :return None:
        """
        model = autodoclass.Autodoclass().train(dummy_folder)
        document_predictions, lines_predictions = model.predict(dummy_text)
        assert isinstance(document_predictions[0], int)
        assert isinstance(document_predictions[1], float)
        for line_prediction in lines_predictions:
            assert isinstance(line_prediction[0], int)
            assert isinstance(line_prediction[1], float)
