import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_max_pool as gmp

class GraphSAGENet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=25, output_dim=128, dropout=0.2):
        super(GraphSAGENet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.sage1 = SAGEConv(num_features_xd, output_dim)
        self.sage2 = SAGEConv(output_dim, output_dim * 2)
        self.sage3 = SAGEConv(output_dim * 2, output_dim * 4)
        self.fc_g1 = torch.nn.Linear(output_dim * 4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Protein sequence branch (1D Conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        # Assuming the length of protein sequences is transformed appropriately
        self.conv_xt_1 = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=8)
        # Calculate the output size after convolution assuming the length of the sequence is `L`
        # This will need to be adjusted based on your actual sequence length
        L = 1000  # Example sequence length
        L_out = L - 8 + 1  # Conv1d output size calculation without padding and stride=1
        self.fc1_xt = nn.Linear(n_filters * L_out, output_dim)

        # Combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # Process graph data
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        # Graph layers
        x = self.sage1(x, edge_index)
        x = self.relu(x)
        x = self.sage2(x, edge_index)
        x = self.relu(x)
        x = self.sage3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)  # Global max pooling

        # Flatten and pass through fully connected layers
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # Protein sequence processing
        embedded_xt = self.embedding_xt(target)
        embedded_xt = embedded_xt.permute(0, 2, 1)  # Adjust shape for Conv1d (batch_size, channels, length)
        conv_xt = self.conv_xt_1(embedded_xt)
        conv_xt = conv_xt.view(conv_xt.size(0), -1)  # Flatten
        xt = self.fc1_xt(conv_xt)

        # Concatenate and process through combined layers
        xc = torch.cat((x, xt), dim=1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out