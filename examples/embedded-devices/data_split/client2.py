import torch
import argparse
import warnings
import flwr as fl
from tqdm import tqdm
from preprocess2 import *
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from torchvision.models import mobilenet_v3_small

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="localhost:8088",
    help=f"gRPC server address (default 'localhost:8088')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)

warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 2
VAL_RATIO = 0.1

# a config for mobilenetv2 that works for
# small input sizes (i.e. 32x32 as in CIFAR)
mb2_cfg = [
    (1, 16, 1, 1),
    (6, 24, 2, 1),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
]


def label_counter(data):
    label_counts = defaultdict(int)

    for datum in data:
        label = datum[1]
        label_counts[label] += 1

    return dict(label_counts)


def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()


def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(device))
            labels = labels.to(device)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


# Flower client, adapted from Pytorch quickstart/simulation example
class FlowerClient(fl.client.NumPyClient):
    """A FlowerClient that trains a MobileNetV3 model for CIFAR-10 or a much smaller CNN
    for MNIST."""

    def __init__(self, trainset, valset, use_mnist):
        self.trainset = trainset
        self.valset = valset
        # Instantiate model
        if use_mnist:
            self.model = Net()
        else:
            self.model = mobilenet_v3_small(num_classes=10)
            # let's not reduce spatial resolution too early
            self.model.features[0][0].stride = (1, 1)
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def set_parameters(self, params):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        print("\nClient sampled for fit()")
        self.set_parameters(parameters)
        # Read hyperparameters from config set by the server
        batch, epochs = config["batch_size"], config["epochs"]
        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        # Define optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # Train
        train(self.model, trainloader, optimizer, epochs=epochs, device=self.device)
        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("\nClient sampled for evaluate()")
        self.set_parameters(parameters)
        # Construct dataloader
        valloader = DataLoader(self.valset, batch_size=64)
        # Evaluate
        loss, accuracy = test(self.model, valloader, device=self.device)
        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

        # Evaluate accuracy on test data
        '''
        testloader = DataLoader(self.testset, batch_size=64)
        loss, accuracy = test(self.model, testloader, device=self.device)

        # Return statistics
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}
        '''


def main():
    args = parser.parse_args()
    print(args)

    assert args.cid < NUM_CLIENTS

    # Download CIFAR-10 dataset and partition it
    handler = DataHandler(num_clients=NUM_CLIENTS, split_ratio=(VAL_RATIO), client_id=args.cid)
    trainset, valset, test = handler.unbalance(noise_level=args.cid / 10, shuffle=True)

    label_counts_tr = label_counter(trainset)

    print("\n---------------TRAIN---------------")
    for label, count in label_counts_tr.items():
        print(f"Label {label}: {count} occurrences")

    label_counts_test = label_counter(valset)

    print("\n---------------VALID---------------")
    for label, count in label_counts_test.items():
        print(f"Label {label}: {count} occurrences")

    # Start Flower client setting its associated data partition
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainset, valset=valset, use_mnist=False
        ),
    )


if __name__ == "__main__":
    main()
