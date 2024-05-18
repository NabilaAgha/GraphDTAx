import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp, GPSConv, MessagePassing

class MyMessagePassing(MessagePassing):
    def __init__(self):
        super(MyMessagePassing, self).__init__(aggr='add')
 
    def forward(self, x, edge_index):
        # Perform message passing operation
        return self.propagate(edge_index, x=x)
 
    def message(self, x_j):
        # The message is simply the input feature x_j
        return x_j
 
    def update(self, aggr_out):
        # The update function does not modify the aggregation result
        return aggr_out

class GCNNet(nn.Module):
    def __init__(self, num_features_xd=78, num_features_xt=25, hidden_channels=78,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2,
                 n_output=1):
        super(GCNNet, self).__init__()
        dim = num_features_xd  # Update dim to match num_features_xd

        # Define the message passing layers
        self.conv1 = MyMessagePassing()
        self.conv2 = MyMessagePassing()
        self.conv3 = MyMessagePassing()

        # Initialize GPSConv using the message passing layers
        self.gps_conv1 = GPSConv(num_features_xd, self.conv1)
        self.gps_conv2 = GPSConv(dim, self.conv2)  # Use dim here
        self.gps_conv3 = GPSConv(dim, self.conv3)  # Use dim here
        self.fc_g1 = nn.Linear(dim, output_dim)

        # Protein Sequence Branch
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(n_filters * 993, output_dim)

        # Combined Layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, n_output)

        # Activation and Regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(dim)  # Update batch normalization layer

    def forward(self, data):
        # Graph Input
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gps_conv1(x, edge_index)
        x = self.relu(x)

        x = self.gps_conv2(x, edge_index)
        x = self.relu(x)
        x = self.batch_norm1(x)  # Apply batch normalization

        x = self.gps_conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)  # Global max pooling

        # Graph Branch
        x = self.fc_g1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Protein Sequence Input
        target = data.target
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt.permute(0, 2, 1))
        xt = conv_xt.view(conv_xt.size(0), -1)

        # Protein Sequence Branch
        xt = self.fc1_xt(xt)
        xt = self.relu(xt)

        # Concatenation
        xc = torch.cat((x, xt), dim=1)

        # Combined Layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        out = self.out(xc)

        return out