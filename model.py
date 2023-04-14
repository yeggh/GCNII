# Definition the CGNII model

class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features):

        super(GCNLayer, self).__init__() 
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(init.kaiming_uniform_(torch.empty(self.in_features, self.out_features), mode='fan_in', nonlinearity='relu'))

    def forward(self, input, adj , h_0 , lamda, alpha, l):

        h_l = torch.spmm(adj, input)
        features = (1 - alpha) * h_l + alpha * h_0
        n = self.weight.shape[0]
        I_n = torch.eye(n) 
        beta = np.log((lamda / l) + 1)
        term1 = (1 - beta) * I_n
        term2 = beta * self.weight
        weights = term1 + term2
        output = torch.mm(features, weights)
        return output

class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha):
        super(GCNII, self).__init__()
        self.graph_convs = nn.ModuleList()  
        for i in range(nlayers):
            conv_layer = GCNLayer(nhidden, nhidden)
            self.graph_convs.append(conv_layer)

        self.pre_fc = nn.Linear(nfeat, nhidden)
        self.post_fc = nn.Linear(nhidden, nclass)
        
        self.relu = nn.ReLU()
        self.dropout = dropout
        self.lamda = lamda
        self.alpha = alpha

    # Forward pass accepts edge_index and edge_attr
    def forward(self, x, edge_index, edge_attr):
        # Construct the adjacency matrix from edge_index and edge_attr
        adj = adjacency_matrix(edge_index, edge_attr)
        adj = normalize_adjacency_matrix(adj)
        adj = sparse_matrix_to_torch_sparse_tensor(adj)

        x = F.dropout(x, self.dropout, training=self.training)
        h_0 = self.relu(self.pre_fc(x))
        h = h_0
        for i, con in enumerate(self.graph_convs):
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.relu(con(h, adj, h_0, self.lamda, self.alpha, i + 1))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.post_fc(h)
        return F.log_softmax(h, dim=1)
    


if __name__ == '__main__':
    pass