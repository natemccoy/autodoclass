import os
from datetime import datetime
from autodoclass import autodoclass


def train(
        args,
        model_filename = "autodoclass.model.{}.pkl".format(
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    ):
    """ Train a model and save the output to a model filename

    :return None:
    """
    model = autodoclass.Autodoclass()
    print("Training model", model, "from folder", args.folder)
    model.train(args.folder)
    print("Saving trained model to filename", model_filename)
    model.save(model_filename)


def predict(args):
    """ Predict from folder and print to stdout using the trained model file

    :return None:
    """
    model = autodoclass.Autodoclass.load(args.model_filename)

    for filename in os.listdir(args.folder):
        path = os.path.join(args.folder, filename)

        with open(path) as f:
            document_text = f.read()

        document_predictions, lines_predictions = model.predict(document_text)
        total_lines = len(lines_predictions)

        print("Predictions for file", path)
        print("Document class; {}, Document class confidence; {}".format(*document_predictions))

        for line_number, line_prediction in enumerate(lines_predictions):
            print("[{}/{}] Line class; {}, Line class confidence; {}".format(line_number, total_lines, *line_prediction))
