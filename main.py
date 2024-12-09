import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


def metrics(data_loader, model):
    model.eval()
    predicted_labels = []
    target_labels = []
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch
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


class FashionMnistModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)  # Flatten the images, use class attribute
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x  # logits

    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits)  # probabilities

    @staticmethod
    def loss(logits, target_labels):
        return F.cross_entropy(logits, target_labels)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Fashion Mnist Training and Evaluation."
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Mode of operation",
        default="train",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        help="Hidden Size of hidden layers of model",
        default=256,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batcg size of dataset",
        default=64,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Total number of epochs used to train",
        default=10,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate of optimizer",
        default=0.0001,
    )
    args = parser.parse_args()

    # data load
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = torchvision.datasets.FashionMNIST(
        root="~/datasets", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root="~/datasets", train=False, transform=transform, download=True
    )
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )
    num_classes = len(test_dataset.classes)
    input_size = train_dataset[0][0].flatten().shape[0]

    # model initialization
    model = FashionMnistModel(input_size, args.hidden_size, num_classes)

    # operation mode
    if args.mode == "train":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        for num_epoch in range(args.num_epochs):
            model.train()
            epoch_metrics = {"loss": []}
            for batch in tqdm(
                train_loader, desc=f"Epoch {num_epoch} | Batch Processing"
            ):
                images, labels = batch
                logits = model(images)
                loss = model.loss(logits, labels)

                # update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_metrics["loss"].append(loss.item())

            # normalize epoch loss
            epoch_metrics["loss"] = np.mean(epoch_metrics["loss"])

            # estimate validation metrics
            val_metrics = metrics(val_loader, model)

            # log metrics
            print(
                f"Epoch #{num_epoch}",
                f"Train Loss: {epoch_metrics['loss']}",
                f"Val Loss: {val_metrics['loss']}",
                f"Val Accuracy: {val_metrics['accuracy']}",
            )

            # save model
            torch.save(model.state_dict(), "fashion_mnist.pth")

    elif args.mode == "eval":
        # evaluation code
        model.eval()
        model.load_state_dict(torch.load("fashion_mnist.pth"))
        test_metrics = metrics(test_loader, model)
        print(
            f"Test Loss: {test_metrics['loss']}",
            f"Test Accuracy: {test_metrics['accuracy']}",
        )

    else:
        raise ValueError(f"{args.mode} is invalid.")


if __name__ == "__main__":
    main()
