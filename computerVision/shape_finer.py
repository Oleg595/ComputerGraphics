import argparse
from read_config import Coordinates, Figure, read_config
from processing_image import get_figures

def parse_args() -> argparse.Namespace:
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("-s", help="input text file")
    args_parser.add_argument("-i", help="input image file")
    namespace = args_parser.parse_args()
    assert namespace.s is not None
    assert namespace.i is not None
    return namespace

def main(args: argparse.Namespace) -> None:
    figures = read_config(args.s)
    contours = get_figures(args.i)

if __name__ == '__main__':
    main(parse_args())
