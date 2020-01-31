import pytest


@pytest.fixture
def dummy_text():
    """ Return a dummy text to use for tests

    :return str: lorem ipsum dummy text
    """
    return """Lorem ipsum dolor sit amet, consectetur adipiscing elit,
    sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
    nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor
    in reprehenderit in voluptate velit esse cillum dolore eu
    fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident,
    sunt in culpa qui officia deserunt mollit anim id est laborum."""


@pytest.fixture
def dummy_folder(tmp_path, dummy_text):
    """ Create a dummy folder used for tests that is populated with
    the dummy text:

    :return str: folder path
    """
    folder = tmp_path / "texts"
    folder.mkdir()
    for file_index in range(10):
        path = folder / "dummy_{file_index}.txt".format(file_index = file_index)
        path.write_text(dummy_text)
    return str(folder)
