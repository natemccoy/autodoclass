import pytest
from autodoclass import clusterer
from autodoclass import encoder


class TestClusterer:
    def test_init(self):
        """ Test initialization the class

        :return None:
        """
        assert clusterer.Clusterer(encoder.Encoder())

    def test_train(self, dummy_folder):
        """ Test train on the base class
        should raise NotImplementedError

        :return None:
        """
        with pytest.raises(NotImplementedError):
            clusterer.Clusterer(encoder.Encoder()).train(dummy_folder)

    def test_predict(self, dummy_text):
        """ Test predict on the base class
        should raise NotImplementedError

        :return None:
        """
        with pytest.raises(NotImplementedError):
            clusterer.Clusterer(encoder.Encoder()).predict(dummy_text)


class TestDocumentClusterer:
    def test_init(self):
        """ Test initialization the class

        :return None:
        """
        assert clusterer.DocumentClusterer(encoder.DocumentEncoder())

    def test_train(self, dummy_folder):
        """ Test training the class

        :return None:
        """
        assert clusterer.DocumentClusterer(
            encoder.DocumentEncoder().train(dummy_folder)
        ).train(dummy_folder)

    def test_predict(self, dummy_folder, dummy_text):
        """ Test predict on the class

        :return None:
        """
        assert clusterer.DocumentClusterer(
            encoder.DocumentEncoder().train(dummy_folder)
        ).train(dummy_folder).predict(dummy_text)


class TestLineClusterer:
    def test_init(self):
        """ Test initialization the class

        :return None:
        """
        assert clusterer.LineClusterer(encoder.LineEncoder())

    def test_train(self, dummy_folder):
        """ Test training the class

        :return None:
        """
        assert clusterer.LineClusterer(
            encoder.LineEncoder().train(dummy_folder)
        ).train(dummy_folder)

    def test_predict(self, dummy_folder, dummy_text):
        """ Test predict on the class

        :return None:
        """
        assert clusterer.LineClusterer(
            encoder.LineEncoder().train(dummy_folder)
        ).train(dummy_folder).predict(dummy_text)
