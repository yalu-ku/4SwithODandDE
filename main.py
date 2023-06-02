from core import *
# from unet import UNet
# from train import train_model

import torch


# def main(args):
#     epochs = 30
#     batch_size = 4
#     lr = 1e-5
#     scale = 0.5
#     val = 10.0
#     amp = False
#     classes = 21
#     bilinear = False
#
#     dataset = VOCDataset()
#
#     model = UNet(n_channels=3, n_classes=classes, bilinear=bilinear)
#     model = model.to(memory_format=torch.channels_last)
#
#     # if args.load:
#     #     state_dict = torch.load(args.load, map_location=device)
#     #     del state_dict['mask_values']
#     #     model.load_state_dict(state_dict)
#
#     model.to(device=device)
#     try:
#         train_model(
#             model=model,
#             epochs=epochs,
#             batch_size=batch_size,
#             learning_rate=lr,
#             device=device,
#             img_scale=scale,
#             val_percent=val / 100,
#             amp=amp,
#             dataset=dataset
#         )
#     except torch.cuda.OutOfMemoryError:
#         torch.cuda.empty_cache()
#         model.use_checkpointing()
#         train_model(
#             model=model,
#             epochs=epochs,
#             batch_size=batch_size,
#             learning_rate=lr,
#             device=device,
#             img_scale=scale,
#             val_percent=val / 100,
#             amp=amp,
#             dataset=dataset
#         )



def main(args):
    model = Model(detection=Detection, depth=Depth, args=args, colors=colors)
    model.run()
    save_all(model)


if __name__ == '__main__':
    _args = parse_args()
    print(_args)
    main(_args)
