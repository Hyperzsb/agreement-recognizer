from agreement_recongizer.config import Config
from agreement_recongizer.model import DenseNetBC
import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def train(dataset_path, saving_path):
    """
    Train the CNN model with dataset specified, output the training result and trained model
    """

    # Load config
    config = Config()
    batch_size = int(config.batch_size)
    epoch = int(config.epoch)
    lr = float(config.lr)

    # Choose available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = DenseNetBC().to(device)

    # Load dataset
    train_dataset = LabeledDataset(dataset_path)
    train_dataset_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr, betas=(0.9, 0.999))
    # Define loss function
    loss = nn.CrossEntropyLoss()
    loss_count = []

    # Start training
    with open(saving_path, "a") as f:
        for epoch in range(epoch):
            for i, train_dataset in enumerate(train_dataset_loader):
                if i == 200:
                    break
                x, y = train_dataset
                x = x.view(x.size(0), 1, 28, 28)
                x = x.float()
                x = x.to(device)
                y = y.to(device)
                output = cnn(x)
                cnn.train()
                # Clear previous grad
                optimizer.zero_grad()
                # Get loss
                loss = loss(output, y)
                # Loss backward
                loss.backward()
                iter_loss = loss.item()
                # Update optimizer arguments
                optimizer.step()
                _, prediction_label = torch.max(output.data, 1)
                correct = (prediction_label == y).sum()
                iter_total = y.size(0)
                f.write('epoch:%03d | iter:%05d | loss:%.03f | acc:%.3f\n'
                        % (epoch + 1, i + 1, float(iter_loss), float(correct) / iter_total))
                f.flush()
                if i > 0 and i % 20 == 19:
                    loss_count.append(loss)
                    torch.save(cnn.state_dict(), './DenseNet.pt')


class LabeledDataset(data.Dataset):
    __image = None
    __label = None

    def __init__(self, location):
        data_train = pd.read_csv(location, index_col=None)
        self.__label = data_train['label']
        data_train.drop('label', axis=1, inplace=True)
        self.__image = data_train.values

    def __getitem__(self, index):
        return self.__image[index], self.__label[index]

    def __len__(self):
        return len(self.__image)
