import pytest
from autodoclass import reader


class TestReader:
    def test_empty_init(self):
        """ Test the initialization of the reader with no arguments

        :return None:
        """
        with pytest.raises(Exception):
            reader.Reader()

    def test_init(self, dummy_folder):
        """ Test the initialization of the reader

        :return None:
        """
        reader.Reader(dummy_folder)

    def test_yield_paths(self, dummy_folder):
        """ Test yielding the paths from the reader

        :return None:
        """
        for path in reader.Reader(dummy_folder).yield_paths():
            assert path


class TestDocumentReader:
    def test_init(self, dummy_folder):
        """ Test the initialization of the DocumentReader class

        :return None:
        """
        reader.DocumentReader(dummy_folder)

    def test_yield_texts(self, dummy_folder):
        """ Test yielding texts from dummy folder

        :return None:
        """
        for text in reader.DocumentReader(dummy_folder).yield_texts():
            assert text


class TestLineReader:
    def test_init(self, dummy_folder):
        """ Test the initialization of the LineReader class

        :return None:
        """
        reader.LineReader(dummy_folder)

    def test_yield_texts(self, dummy_folder):
        """ Test yielding texts from dummy folder

        :return None:
        """
        for text in reader.LineReader(dummy_folder).yield_texts():
            assert text

    def test_yield_texts_no_line_endings(self, dummy_folder):
        """ Test yielding texts from dummy folder

        :return None:
        """
        for text in reader.LineReader(dummy_folder).yield_texts():
            assert not text.endswith('\n')   # unix-like
            assert not text.endswith('\r\n') # windows-like
