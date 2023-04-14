# GCNII
PyTorch Implementation of the Paper Simple and Deep Graph Convolutional Networks

Graph Convolutional Networks (GCNs) are innovative extensions of Convolutional Neural Networks (CNNs) that enable the processing of graph-structured data, e.g., social or traffic road networks. GCNs use "graph convolution" to learn graph representations by applying a linear transformation and a nonlinear activation function in the local neighbourhood of each node. As a result, deepening a GCN can lead to over-smoothing, where the representation of all nodes becomes increasingly similar as the number of layers increases. Therefore, it is essential to carefully consider the depth of the network when using GCNs for analysis or prediction. To address this issue, our selected paper proposes GCNII [^1^], a method for deepening GCN models -- in particular the Vanilla GCN [^2^] -- using initial residual connections and identity mapping. The initial residual connection is similar to the skip connection in ResNet [^3^], and identity mapping preserves the input values of a neural network layer. These methods help the network retain critical information from the input as it is processed through subsequent layers, preventing over-smoothing.

[^1^]: [GCNII](https://dblp.org/rec/conf/icml/ChenWHDL20)
[^2^]: [Vanilla GCN](https://arxiv.org/abs/1609.02907)
[^3^]: [ResNet](https://dblp.org/rec/conf/cvpr/HeZRS16)

# Code

To run the semi-supervised code refer to 'GCNII_semi_supervised.ipynb'.

To run the full-supervised for datasets Cora, Citeseer, and PubMed code refer to 'GCNII_full_supervised_citation.ipynb'.

To run the full-supervised for datasets Chameleon, Cornell, Texas, and Wisconsincode refer to 'GCNII_full_supervised-WebKb.ipynb'.
