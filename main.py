from core import *


def main(args):
    model = Model(detection=Detection, depth=Depth, args=args, colors=colors)
    model.run()
    # save_all(model)


if __name__ == '__main__':
    _args = parse_args()
    print(_args)
    main(_args)
