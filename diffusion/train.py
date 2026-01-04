from argparse import ArgumentParser
from utils import parse_unknown_args, get_num_classes, load_model, save_model
from utils.process_data import load_TensorDataset, load_NIST

from diffusion.architecture import DiffusionCNN, DiffusionViT
from diffusion.loss import train, PlFlow


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["MNIST", "EMNIST", "FashionMNIST"], required=True)
    parser.add_argument("--encoded", action="store_true", help="If provided, use the latent dataset")
    parser.add_argument("--split", choices=["balanced", "byclass", "bymerge"], default="balanced")
    parser.add_argument("--data-dir", default="data")

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--model", choices=["cnn", "vit"], required=True)
    parser.add_argument("--load", choices=["dev", "main"], default=None)
    parser.add_argument("--noise-type", choices=["gaussian", "uniform", "bernoulli"], default="gaussian")

    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--total-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--milestones", type=int, nargs="*", default=[15, 25, 35, 45])
    parser.add_argument("--gamma", type=float, default=0.5, help="Decay of lr at each milestone")

    args, unknown = parser.parse_known_args()
    return args, unknown


def main(args, **nn_kwargs):
    # dataset
    if args.encoded:
        train_loader = load_TensorDataset(
            root=args.data_dir, data_path=f"{args.dataset}_encoded",
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
    else:
        train_loader = load_NIST(
            dataset=args.dataset, train=True,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )

    # model
    arch = DiffusionCNN if args.model == "cnn" else DiffusionViT
    model = load_model(
        model_architecture=arch, dataset=args.dataset,
        model_version=args.load, **nn_kwargs,
        n_classes=get_num_classes(args.dataset, args.split),
    ).to(args.device)

    # train loop
    train(
        model=model,
        pl_class=PlDiffusion,
        dataset=args.dataset,
        train_loader=train_loader,
        lr=args.lr,
        total_epochs=args.total_epochs,
        milestones=args.milestones,
        gamma=args.gamma,
        test_loader=None,
        noise_type=args.noise_type,
    )
    ans = input(f"Save this model as the main {args.model} model? [y/N]: ")
    if ans.lower() == "y":
        save_model(model, dataset=args.dataset, model_version="main")


if __name__ == "__main__":
    args, unknown = parse_args()
    nn_kwargs = parse_unknown_args(unknown)
    main(args, **nn_kwargs)
