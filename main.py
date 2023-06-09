from core import Model
from utils import *


def main(args):
    model = Model(detection=Detection, depth=Depth, args=args, colors=colors)
    model.inference()
    model.preprocess()
    model.crop()
    model.postprocess()
    model.ordered_paint()
    # save_image(model.gray_image, 'gray_image.png')
    # save_image(model.blur_image, 'blur_image.png')
    # save_image(model.thresh_image, 'thresh_image.png')
    # save_image(model.edge, 'edge.png')
    # save_image(model.closed_edge_image, 'closed_edge_image.png')
    # save_image(model.contours_image, 'contours_image.png')
    save_image(model.processing_image, 'normalized_depth_output.png')
    save_image(model.ordered_image, 'ordered_image.png')
    save_image(model.masking_image, 'masking_image.png')
    save_images(model.names, model.crop_images)


if __name__ == '__main__':
    _args = parse_args()
    print(_args)
    main(_args)
