import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool

class EnhancedGAT(nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 hidden_channels=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(EnhancedGAT, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        self.output_dim = output_dim  # Store output_dim as an instance variable

        # Drug branch
        self.gat1 = GATConv(num_features_xd, hidden_channels, heads=4, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, dropout=dropout)
        self.gat3 = GATConv(hidden_channels * 4, hidden_channels, heads=4, dropout=dropout)
        self.gat4 = GATConv(hidden_channels * 4, hidden_channels, heads=4, dropout=dropout)
        self.gat5 = GATConv(hidden_channels * 4, hidden_channels, heads=4, dropout=dropout)
        self.fc1_xd = nn.Linear(hidden_channels * 4, self.output_dim)

        # Target branch
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=embed_dim, out_channels=hidden_channels, kernel_size=8)
        self.num_features_xt = None  # To be calculated after the first forward pass
        self.fc1_xt = None  # To be initialized after the first forward pass

        # Combined branches
        self.fc1 = nn.Linear(self.output_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        # Drug branch GAT layers
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        x = F.relu(self.gat3(x, edge_index))
        x = F.relu(self.gat4(x, edge_index))
        x = F.relu(self.gat5(x, edge_index))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = self.dropout(x)

        # Target branch embedding and convolution
        embedded_xt = self.embedding_xt(target)
        embedded_xt = embedded_xt.permute(0, 2, 1)
        conv_xt = self.conv_xt_1(embedded_xt)

        # On the first forward pass, determine the number of features and initialize the fc1_xt layer
        if self.fc1_xt is None:
            self.num_features_xt = conv_xt.size(1) * conv_xt.size(2)
            self.fc1_xt = nn.Linear(self.num_features_xt, self.output_dim).to(conv_xt.device)

        # Flatten the output of the convolution
        xt = conv_xt.view(conv_xt.size(0), -1)
        xt = F.relu(self.fc1_xt(xt))

        # Combine drug and target branch outputs
        xc = torch.cat((x, xt), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out