import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, y_dim, c_dim, img_size):
        """
        Initialize classifier
        Inputs:
        - y_dim : number of classes
        """
        stride = 1

        # Magic formula to account for result of shape change: ((img_size - 2stride - 2stride) / 2) ** 2) * 64.
        # Better than hard coding 12544 or 9216 for 32 or 28 -sized images respectively
        hidden_size = int((((img_size - 4 * stride) / 2) ** 2) * 64)

        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(c_dim, 32, 3, stride)
        self.conv2 = nn.Conv2d(32, 64, 3, stride)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, y_dim)

    def forward(self, x):
        """
        Perform classification using the CNN classifier

        Inputs:
        - x : input data sample

        Outputs:
        - out: unnormalized output
        - prob_out: probability output
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        out = self.fc2(x)
        prob_out = F.softmax(out)

        return prob_out, out