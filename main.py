from core import Model
from utils import *


def main(args):
    model = Model(detection=Detection, depth=Depth, args=args)
    model.inference()
    model.crop()
    save_images(model.names, model.crop_images, args.source)


if __name__ == '__main__':
    _args = parse_args()
    print(_args)
    main(_args)
