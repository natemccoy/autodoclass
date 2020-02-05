import argparse
from autodoclass import commands


def main():
    """ Main function call, parse arguments and run commands

    :return None:
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('folder')
    train_parser.set_defaults(callback = commands.train)

    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('folder')
    predict_parser.add_argument('model_filename')
    predict_parser.set_defaults(callback = commands.predict)

    args = parser.parse_args()

    if not hasattr(args, 'callback'):
        parser.print_help()
        exit()

    args.callback(args)


if __name__ == "__main__":
    main()
