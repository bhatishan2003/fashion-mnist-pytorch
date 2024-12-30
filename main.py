import argparse
import csv
import os
import random

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


def evaluate(data_loader, model, device="cpu"):
    model.eval()
    predicted_labels = []
    target_labels = []
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = model.loss(logits, labels)

            # estimate metrics
            predicted_labels += torch.argmax(logits, dim=1).tolist()
            target_labels += labels.tolist()
            losses.append(loss.item())

    return {
        "loss": np.mean(losses),
        "accuracy": np.mean(np.array(predicted_labels) == np.array(target_labels)),
    }


class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation):
        super().__init__()
        self.hidden_Size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.activation = activation

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)  # Flatten the images
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x  # logits

    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)  # probabilities

    @staticmethod
    def loss(logits, target_labels):
        return F.cross_entropy(logits, target_labels)


class FashionMNISTCNN(nn.Module):
    def __init__(self, input_channels, num_classes, activation):
        super(FashionMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.activation(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)  # probabilities

    @staticmethod
    def loss(logits, target_labels):
        return F.cross_entropy(logits, target_labels)


def get_activation_function(activation_name):
    if activation_name in ["relu", "leaky_relu", "tanh", "sigmoid"]:
        return getattr(F, activation_name)
    else:
        raise ValueError(f"Unsupported activation function entered: {activation_name}")


def configure_optimizer(optimizer_name, model_parameters, learning_rate):
    if optimizer_name == "sgd":
        return torch.optim.SGD(model_parameters, lr=learning_rate)
    elif optimizer_name == "adam":
        return torch.optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model_parameters, lr=learning_rate)
    elif optimizer_name == "adagrad":
        return torch.optim.Adagrad(model_parameters, lr=learning_rate)
    else:
        raise ValueError(f"Incorrect optimizer entered: {optimizer_name}")


def get_args():
    parser = argparse.ArgumentParser(description="FashionMNIST Training and Evaluation.")
    parser.add_argument("--project_name", type=str, default="fashion-mnist-pytorch", help="Name of project")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode of operation")
    parser.add_argument("--seed", type=int, default=10, help="Seed to control randomness")
    parser.add_argument("--model_type", type=str, default="fc", choices=["cnn", "fc"], help="Choice of model")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size (for FC model only)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--disable_normalization", action="store_true", help="Disable dataset normalization", default=False)
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "leaky_relu", "tanh", "sigmoid"],
        help="Activation function for network",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["sgd", "adam", "rmsprop", "adagrad"], help="Optimizer"
    )
    parser.add_argument("--experiment_root_dir", type=str, default=os.path.join(os.getcwd(), "results"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If enabled and existing checkpoint exits, the experiment is resumed.",
        default=False,
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable Cuda/GPU usage", default=False)
    parser.add_argument("--use_mlflow", action="store_true", help="If enabled, uses mlflow for logging", default=False)
    parser.add_argument("--mlflow_tracking_uri", type=str, default="file:./mlruns")
    args = parser.parse_args()

    # create run directory
    run_name = f"seed-{args.seed}-model_type-{args.model_type}-hidden_size-{args.hidden_size}"
    run_name += f"-batch_size-{args.batch_size}-num_epochs-{args.num_epochs}-learning_rate-{args.learning_rate}"
    run_name += f"-disable_normalization-{args.disable_normalization}-activation-{args.activation}-optimizer-{args.optimizer}"
    args.run_dir = os.path.join(args.experiment_root_dir, run_name)
    os.makedirs(args.run_dir, exist_ok=True)

    # model-paths
    args.checkpoint_path = os.path.join(args.run_dir, "checkpoint.pth")

    # device selection
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    return args


def main():
    # Command line arguments for experiment
    args = get_args()

    # Seed randomness
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Data transformation operation
    transform_operations = [transforms.ToTensor()]
    if not args.disable_normalization:
        transform_operations.append(transforms.Normalize((0.5,), (0.5,)))
    transform = transforms.Compose(transform_operations)

    # Data download and loader creation
    train_dataset = torchvision.datasets.FashionMNIST(root="~/datasets", train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root="~/datasets", train=False, transform=transform, download=True)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    num_classes = len(test_dataset.classes)

    # Model selection
    activation = get_activation_function(args.activation)
    if args.model_type == "cnn":
        model = FashionMNISTCNN(input_channels=1, num_classes=num_classes, activation=activation)
    elif args.model_type == "fc":
        model = FullyConnectedNN(
            input_size=train_dataset[0][0].flatten().shape[0],
            hidden_size=args.hidden_size,
            num_classes=num_classes,
            activation=activation,
        )
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    # Mode Operation
    if args.mode == "train":
        optimizer = configure_optimizer(args.optimizer, model.parameters(), args.learning_rate)

        # setup mlflow
        if args.use_mlflow:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
            mlflow.set_experiment(args.project_name)
            mlflow.start_run()
            mlflow.log_params(
                {
                    "seed": args.seed,
                    "model_type": args.model_type,
                    "hidden_size": args.hidden_size,
                    "batch_size": args.batch_size,
                    "num_epochs": args.num_epochs,
                    "learning_rate": args.learning_rate,
                    "disable_normalization": args.disable_normalization,
                    "activation": args.activation,
                    "optimizer": args.optimizer,
                }
            )

        # logging utility: we write our logs in a csv file
        logs_path = os.path.join(args.run_dir, "logs.csv")
        if os.path.exists(logs_path):
            csv_logger_file = open(logs_path, "a", newline="")
            csv_logger_writer = csv.writer(csv_logger_file)
        else:
            csv_logger_file = open(logs_path, "w", newline="")
            csv_logger_writer = csv.writer(csv_logger_file)
            csv_logger_writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])

        # resume experiment
        epoch_start_num = 0
        if args.resume:
            if os.path.exists(args.checkpoint_path):
                checkpoint_dict = torch.load(args.checkpoint_path)
                model.load_state_dict(checkpoint_dict["model"])
                optimizer.load_state_dict(checkpoint_dict["optimizer"])
                epoch_start_num = checkpoint_dict["epoch"] + 1
            else:
                print("No existing checkpoint found! Start experiment from scratch.")

        # device assignment
        model = model.to(args.device)

        # epoch training
        for epoch in range(epoch_start_num, args.num_epochs):
            model.train()
            epoch_metrics = {"train-loss": []}

            with tqdm(total=len(train_loader)) as pbar:
                pbar.set_description(f"Training Epoch: {epoch} | Batch Processing:")
                for batch in train_loader:
                    # batch formulate and forward
                    images, labels = batch
                    images = images.to(args.device)
                    labels = labels.to(args.device)
                    logits = model(images)

                    # Update the model
                    loss = model.loss(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_metrics["train-loss"].append(loss.item())

                    # update progress bat
                    pbar.update(1)

                # normalize
                epoch_metrics["train-loss"] = np.mean(epoch_metrics["train-loss"])

                # Validation
                for k, v in evaluate(val_loader, model, args.device).items():
                    epoch_metrics[f"val-{k}"] = v

                # Logging metrics
                if args.use_mlflow:
                    mlflow.log_metrics({**epoch_metrics, "epoch": epoch}, step=epoch)
                msg = f"Training Epoch: {epoch} |"
                msg += ",".join(f"{k}: {round(v,2)}" for k, v in epoch_metrics.items())
                msg += " | Batch Processing "
                pbar.set_description(msg)

                # write metrics to csv
                csv_logger_writer.writerow(
                    [epoch, epoch_metrics["train-loss"], epoch_metrics["val-loss"], epoch_metrics["val-accuracy"]]
                )

            # Save the model
            torch.save({"optimizer": optimizer.state_dict(), "model": model.state_dict(), "epoch": epoch}, args.checkpoint_path)
        # close logging utility
        csv_logger_file.close()
        if args.use_mlflow:
            mlflow.end_run()

    elif args.mode == "eval":
        # Load and evaluate the model
        model.load_state_dict(torch.load(args.checkpoint_path)["model"])
        model = model.to(args.device)
        test_metrics = evaluate(test_loader, model, args.device)
        print(
            f"Test Loss: {test_metrics['loss']:.4f}",
            f"Test Accuracy: {test_metrics['accuracy']:.4f}",
        )

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
