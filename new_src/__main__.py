import argparse
import torch

from new_src.train import GeneralModel, PlAutoEncoder, PlVarAutoEncoder, PlDiffusion, train, test
from new_src.architectures import AutoEncoder, VarAutoEncoder, Diffusion
from new_src.generate import diffusion_generation, autoencoder_reconstruction
from new_src.data import encode_dataset, load_TensorDataset, load_EMNIST, load_MNIST, load_FashionMNIST
from new_src.ml_utils import load_model, save_model


TASK_LOADERS = {
    "MNIST": load_MNIST,
    "FashionMNIST": load_FashionMNIST,
    "EMNIST": load_EMNIST,
}

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
    parser = argparse.ArgumentParser(description="Project CLI")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Common arguments
    def add_common(subp):
        subp.add_argument("--task", choices=TASK_LOADERS.keys(), required=True)
        subp.add_argument("--arch", choices=ARCHS.keys())
        subp.add_argument("--model-version", choices=["dev", "main"], default="dev")
        subp.add_argument("--parameters-root", default="parameters")
        subp.add_argument("--data-root", default="data")

    # Training
    train_p = subparsers.add_parser("training")
    add_common(train_p)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--total-epochs", type=int, default=10)
    train_p.add_argument("--batch-size", type=int, default=128)
    train_p.add_argument("--num-workers", type=int, default=0)
    train_p.add_argument("--milestones", type=int, nargs="*", default=[])
    train_p.add_argument("--gamma", type=float, default=0.1)

    # Testing
    test_p = subparsers.add_parser("testing")
    add_common(test_p)
    test_p.add_argument("--batch-size", type=int, default=128)

    # Encode dataset
    dp_p = subparsers.add_parser("encode-dataset")
    add_common(dp_p)

    # Generation
    gen_p = subparsers.add_parser("generation")
    add_common(gen_p)

    # Reconstruction
    rec_p = subparsers.add_parser("reconstruction")
    add_common(rec_p)

    args, unknown = parser.parse_known_args()
    return args, unknown


def parse_unknown_args(unknown) -> dict[str, int]:
    """Pass all unknown arguments into the NN"""
    nn_kwargs = {}
    for i in range(0, len(unknown), 2):
        key = unknown[i].lstrip("-").replace("-", "_")
        val = unknown[i + 1]
        nn_kwargs[key] = int(val)
    return nn_kwargs


def main(args, **nn_kwargs):
    task = args.task
    loader_fn = TASK_LOADERS[task]

    if args.mode in ["training", "testing"]:
        pl_class = PLARCHS[args.arch]

        if args.arch == "flow":
            data_path = f"{task}_encoded"
            train_loader = load_TensorDataset(
                root=args.data_root, data_path=data_path, shuffle=True,
                batch_size=args.batch_size, num_workers=args.num_workers,
            )
            test_loader = load_TensorDataset(
                root=args.data_root, data_path=data_path,
                shuffle=False, batch_size=args.batch_size,
            )
        else:
            train_loader = loader_fn(
                train=True, batch_size=args.batch_size, num_workers=args.num_workers
            )
            test_loader = loader_fn(train=False, batch_size=args.batch_size)

        model = load_model(
            model_architecture=pl_class.model_architecture,
            model_version=args.model_version,
            root=args.parameters_root,
            **nn_kwargs,
        )

        if args.mode == "training":
            train(
                model=model,
                pl_class=pl_class,
                train_loader=train_loader,
                test_loader=test_loader,
                lr=args.lr,
                total_epochs=args.total_epochs,
                milestones=args.milestones,
                gamma=args.gamma,
            )

            ans = input(f"Save this {args.arch} model as '{args.arch}.pth'? [y/N]: ")
            if ans.lower() == "y":
                save_model(model, model_version="main", root=args.parameters_root)

        elif args.mode == "testing":
            test(
                model=model,
                pl_class=pl_class,
                test_loader=test_loader,
            )

    elif args.mode == "encode-dataset":
        autoencoder = load_model(
            model_architecture=ARCHS[args.arch],
            model_version=args.model_version,
            root=args.parameters_root,
            **nn_kwargs,
        )
        data = loader_fn(train=True)
        encode_dataset(
            data, autoencoder, save_path=f"{task}_encoded",
            root=args.data_root, batch_size=args.batch_size,
        )

    elif args.mode == "generation":  # TODO: load two models
        model = load_model(
            Diffusion, model_version=args.model_version, root=args.parameters_root
        )
        autoencoder = load_model(
            ARCHS[args.arch], model_version=args.model_version, root=args.parameters_root
        )
        diffusion_generation(model, autoencoder)

    elif args.mode == "reconstruction":
        autoencoder = load_model(
            ARCHS[args.arch], model_version=args.model_version,
            root=args.parameters_root, **nn_kwargs,
        )
        dataloader = loader_fn(train=False)
        autoencoder_reconstruction(autoencoder, dataloader)


if __name__ == "__main__":
    args, unknown = parse_args()
    nn_kwargs = parse_unknown_args(unknown)
    main(args, **nn_kwargs)
