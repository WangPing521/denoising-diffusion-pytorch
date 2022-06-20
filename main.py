import argparse

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.helper import mask_transform, png_transform


def get_args():
    parser = argparse.ArgumentParser(
        description="diffusion model for cifar and segmentation mask",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        required=True,
        choices=["cifar10", "seg_mask"],
        help="data set",
    )
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="batch size")
    parser.add_argument("--image-size", "-s", type=int, default=96, help="image size")
    args = parser.parse_args()
    if args.data == "cifar10":
        args.input_channels = 3
        args.output_channels = 3
        args.transform = png_transform(args.image_size)
        args.data_path = "./cifar10"
    else:
        args.input_channels = 4
        args.output_channels = 4
        args.transform = mask_transform(image_size=args.image_size, num_classes=4)
        args.data_path = "./seg_mask"

    return args


def main(args):
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=args.input_channels,
        out_dim=args.output_channels,
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=1000,
        loss_type="l1",  # number of steps  # L1 or L2
    ).cuda()

    trainer = Trainer(
        diffusion,
        args.data_path,
        transform=args.transform,
        train_batch_size=args.batch_size,
        train_lr=1e-4,
        train_num_steps=700000,  # total training steps
        gradient_accumulate_every=1,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
    )

    trainer.train()


if __name__ == "__main__":
    from torch.backends import cudnn

    cudnn.benchmark = True

    args = get_args()
    main(args)
