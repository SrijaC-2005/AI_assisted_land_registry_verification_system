import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class LandFraudGNN(torch.nn.Module):

    def __init__(
        self,
        input_features,
        hidden_channels=64,
        output_classes=2
    ):

        super().__init__()

        self.conv1 = GCNConv(
            input_features,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.classifier = torch.nn.Linear(
            hidden_channels,
            output_classes
        )

        self.dropout = 0.3


    def forward(self, x, edge_index):

        # First GCN layer
        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        # Second GCN layer
        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        # Classification
        x = self.classifier(x)

        return x