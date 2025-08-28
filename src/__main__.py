from argparse import ArgumentParser
import torch

from ml_train import GeneralModel, PlAutoEncoder, PlVarAutoEncoder, PlDiffusion, train, test
from architectures import AutoEncoder, VarAutoEncoder, Diffusion
from generate import diffusion_generation, autoencoder_reconstruction
from process_data import encode_dataset, load_TensorDataset, load_NIST
from ml_utils import load_model, save_model


ARCHS: dict[str, type[torch.nn.Module]] = {
    "autoencoder": AutoEncoder,
    "vae": VarAutoEncoder,
    "flow": Diffusion,
}

PLARCHS: dict[str, type[GeneralModel]] = {
    "autoencoder": PlAutoEncoder,
    "vae": PlVarAutoEncoder,
    "flow": PlDiffusion,
}


def parse_args():
    parser = ArgumentParser(description="Project CLI")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Common arguments
    def add_common(
        subp: ArgumentParser,
        version_default: str | None = "main"
    ):
        subp.add_argument("--dataset", choices=["MNIST", "EMNIST", "FashionMNIST"], required=True)
        subp.add_argument("--arch", choices=ARCHS.keys())
        subp.add_argument("--model-version", choices=["dev", "main"], default=version_default)
        subp.add_argument("--root", default="data")
        subp.add_argument("--batch-size", type=int, default=128)
        subp.add_argument("--split", choices=["balanced", "byclass", "bymerge"], default="balanced")

    # Plot arguments
    def add_plot(subp: ArgumentParser):
        subp.add_argument("--height", type=int, default=10)
        subp.add_argument("--width", type=int, default=10)
        subp.add_argument("--scale", type=float, default=1.0)

    # Training
    train_p = subparsers.add_parser("train")
    add_common(train_p, version_default=None)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--total-epochs", type=int, default=10)
    train_p.add_argument("--num-workers", type=int, default=0)
    train_p.add_argument("--milestones", type=int, nargs="*", default=[])
    train_p.add_argument("--gamma", type=float, default=0.1)

    # Testing
    test_p = subparsers.add_parser("test")
    add_common(test_p)
    test_p.add_argument("--num-workers", type=int, default=0)

    # Encode dataset
    dp_p = subparsers.add_parser("encode-dataset")
    add_common(dp_p)

    # Generation
    gen_p = subparsers.add_parser("generation")
    add_common(gen_p)
    add_plot(gen_p)
    gen_p.add_argument("--weight", type=float, default=1)
    gen_p.add_argument("--diffusion", type=float, default=1, help="level of noise")
    gen_p.add_argument("--autoencoder-version", choices=["dev", "main"], default="main")

    # Reconstruction
    rec_p = subparsers.add_parser("reconstruction")
    add_common(rec_p)
    add_plot(rec_p)

    args, unknown = parser.parse_known_args()
    return args, unknown


def parse_unknown_args(unknown) -> dict[str, int]:
    """
    Pass all unknown arguments into the NN
    This feature is only used in train mode
    """
    nn_kwargs = {}
    for i in range(0, len(unknown), 2):
        key = unknown[i].lstrip("-").replace("-", "_")
        val = unknown[i + 1]
        nn_kwargs[key] = int(val)
    return nn_kwargs


def main(args, **nn_kwargs):
    if args.mode in ["train", "test"]:
        pl_class = PLARCHS[args.arch]

        if args.arch == "flow":
            data_path = f"{args.dataset}_encoded"
            if args.mode == "train":
                train_loader = load_TensorDataset(
                    root=args.root, data_path=data_path, shuffle=True,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                )
            test_loader = load_TensorDataset(
                root=args.root, data_path=data_path, shuffle=False,
                batch_size=args.batch_size, num_workers=args.num_workers,
            )
        else:
            if args.mode == "train":
                train_loader = load_NIST(
                    dataset=args.dataset,
                    train=True,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers
                )
            test_loader = load_NIST(
                dataset=args.dataset,
                train=False,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                )

        model = load_model(
            model_architecture=pl_class.model_architecture,
            dataset=args.dataset,
            split=args.split,
            model_version=args.model_version,
            **nn_kwargs,
        )

        if args.mode == "train":
            train(
                model=model,
                pl_class=pl_class,
                dataset=args.dataset,
                train_loader=train_loader,  # type: ignore
                test_loader=test_loader,
                lr=args.lr,
                total_epochs=args.total_epochs,
                milestones=args.milestones,
                gamma=args.gamma,
            )

            ans = input(f"Save this {args.arch} model as '{args.arch}.pth'? [y/N]: ")
            if ans.lower() == "y":
                save_model(model, dataset=args.dataset, model_version="main")

        elif args.mode == "test":
            test(
                model=model,
                pl_class=pl_class,
                dataset=args.dataset,
                test_loader=test_loader,
            )

    else:
        autoencoder = load_model(
            model_architecture=ARCHS[args.arch],
            dataset=args.dataset,
            split=args.split,
            model_version=args.model_version,
        )

        if args.mode == "encode-dataset":
            data = load_NIST(dataset=args.dataset, train=True,)
            encode_dataset(
                data=data,
                autoencoder=autoencoder,  # type: ignore
                save_path=f"{args.dataset}_encoded",
                root=args.root,
                batch_size=args.batch_size,
            )

        elif args.mode == "generation":
            flow = load_model(
                Diffusion,
                dataset=args.dataset,
                split=args.split,
                model_version=args.model_version,
            )
            diffusion_generation(
                flow,         # type: ignore
                autoencoder,  # type: ignore
                labels=[k % 10 for k in range(100)],
                weight=args.weight,
                diffusion=args.diffusion,
                width=args.width,
                height=args.height,
                scale=args.scale,
            )

        elif args.mode == "reconstruction":
            dataloader = load_NIST(dataset=args.dataset, train=False)
            autoencoder_reconstruction(
                autoencoder,  # type: ignore
                dataloader,
                width=args.width,
                height=args.height,
                scale=args.scale,
            )


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args, unknown = parse_args()
    nn_kwargs = parse_unknown_args(unknown)
    main(args, **nn_kwargs)
