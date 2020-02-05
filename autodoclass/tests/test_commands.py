import pytest
import os
import collections
from autodoclass import commands
from autodoclass import autodoclass


def test_commands_train(dummy_folder):
    """ Test the train command

    :return None:
    """
    DummyArgs = collections.namedtuple("DummyArgs", "folder")
    test_model_filename = "test_commands_train_model.pkl"
    args = DummyArgs(folder = dummy_folder)
    commands.train(args, model_filename = test_model_filename)
    os.remove(test_model_filename)


def test_commands_predict(dummy_folder):
    """ Test the predict command

    :return None:
    """
    DummyArgs = collections.namedtuple("DummyArgs", "folder model_filename")
    test_model_filename = "test_commands_predict_model.pkl"
    args = DummyArgs(folder = dummy_folder, model_filename = test_model_filename)
    autodoclass.Autodoclass().train(dummy_folder).save(test_model_filename)
    commands.predict(args)
    os.remove(test_model_filename)
