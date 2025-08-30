print("Importing packages...")
from argparse import ArgumentParser
import torch

from pl_train import GeneralModel, PlAutoEncoder, PlVarAutoEncoder, PlDiffusion
from pl_train import train, test
from architectures import AutoEncoder, VarAutoEncoder, Diffusion
from generate import diffusion_generation, autoencoder_reconstruction
from process_data import encode_dataset, load_TensorDataset, load_NIST, load_extra_args
from ml_utils import load_model, save_model, get_num_classes


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
        version_default: str | None = "main",
        model_choices: list[str] = ["autoencoder", "vae"],
    ):
        subp.add_argument("--dataset", choices=["MNIST", "EMNIST", "FashionMNIST"], required=True)
        subp.add_argument("--model", choices=model_choices, required=True)
        subp.add_argument("--model-version", choices=["dev", "main"], default=version_default)
        subp.add_argument("--root", default="data")
        subp.add_argument("--batch-size", type=int, default=128)
        subp.add_argument("--split", choices=["balanced", "byclass", "bymerge"], default="balanced")
        subp.add_argument("--device", type=str, default="cpu")

    # Plot arguments
    def add_plot(subp: ArgumentParser):
        subp.add_argument("--height", type=int, default=8)
        subp.add_argument("--width", type=int, default=8)
        subp.add_argument("--scale", type=float, default=0.8)

    # Training
    train_p = subparsers.add_parser("train")
    add_common(train_p, version_default=None, model_choices=["autoencoder", "vae", "flow"])
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--total-epochs", type=int, default=10)
    train_p.add_argument("--num-workers", type=int, default=0)
    train_p.add_argument("--milestones", type=int, nargs="*", default=[])
    train_p.add_argument("--gamma", type=float, default=0.2, help="Decay of lr at each milestone")
    train_p.add_argument("--alpha", type=float, default=0.2, help="Weight of KL loss for VAEs")

    # Testing
    test_p = subparsers.add_parser("test")
    add_common(test_p)
    test_p.add_argument("--num-workers", type=int, default=0)
    test_p.add_argument("--autoencoder-version", choices=["dev", "main"], default="main")

    # Encode dataset
    dp_p = subparsers.add_parser("encode-dataset")
    add_common(dp_p)

    # Generation
    gen_p = subparsers.add_parser("generate")
    add_common(gen_p)
    add_plot(gen_p)
    gen_p.add_argument("--weight", type=float, default=4, help="Classifier-free guidance weight")
    gen_p.add_argument("--diffusion", type=float, default=1.5, help="level of noise")
    gen_p.add_argument("--autoencoder-version", choices=["dev", "main"], default="main")

    # Reconstruction
    rec_p = subparsers.add_parser("test-reconstruction")
    add_common(rec_p)
    add_plot(rec_p)

    args, unknown = parser.parse_known_args()
    return args, unknown


def parse_unknown_args(unknown: list[str]) -> dict[str, int]:
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
    """Implements all functionalities of the repo so they can run from the terminal"""
    if args.mode in ["train", "test"]:
        pl_class = PLARCHS[args.model]  # autoencoder class if mode=test

        if args.model == "flow":
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
            nn_kwargs["n_classes"] = get_num_classes(args.dataset, args.split)
            nn_kwargs["z_shape"] = load_extra_args(data_path, args.root, "z_shape.pickle")
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

        if args.mode == "train":
            model = load_model(
                model_architecture=pl_class.model_architecture,
                dataset=args.dataset,
                model_version=args.model_version,
                **nn_kwargs,
            ).to(args.device)
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
                alpha=args.alpha,
            )
            ans = input(f"Save this model as the main {args.model} model? [y/N]: ")
            if ans.lower() == "y":
                save_model(model, dataset=args.dataset, model_version="main")

        elif args.mode == "test":
            pl_class = PLARCHS[args.model]

            flow = load_model(
                Diffusion,
                dataset=args.dataset,
                model_version=args.model_version,
            ).to(args.device)
            autoencoder = load_model(
                pl_class.model_architecture,
                dataset=args.dataset,
                model_version=args.autoencoder_version,
            ).to(args.device)
            test(
                flow=flow,  # type: ignore
                autoencoder=autoencoder,  # type: ignore
                test_loader=test_loader,
            )

    else:
        autoencoder = load_model(
            model_architecture=ARCHS[args.model],
            dataset=args.dataset,
            model_version=args.model_version,
        ).to(args.device)

        if args.mode == "encode-dataset":
            data = load_NIST(dataset=args.dataset, train=True,)
            encode_dataset(
                data=data,
                autoencoder=autoencoder,  # type: ignore
                save_path=f"{args.dataset}_encoded",
                root=args.root,
                batch_size=args.batch_size,
            )

        elif args.mode == "generate":
            flow = load_model(
                Diffusion,
                dataset=args.dataset,
                model_version=args.model_version,
            ).to(args.device)
            n_classes = get_num_classes(args.dataset, args.split)
            diffusion_generation(
                flow,         # type: ignore
                autoencoder,  # type: ignore
                labels=[k % n_classes for k in range(args.height * args.width)],
                weight=args.weight,
                diffusion=args.diffusion,
                width=args.width,
                height=args.height,
                scale=args.scale,
            )

        elif args.mode == "test-reconstruction":
            dataloader = load_NIST(
                dataset=args.dataset,
                train=False,
                batch_size=args.height*args.width,
            )
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
