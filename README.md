# Graph Neural Network

## Available Models

    1. Fully Connected
    2. Graph (with my Custom Adj Matrix - default)
    3. Convolution
    4. Graph (with Gaussian Adj Matrix)

## Implementation of a Graph Neural Network (MNIST) with 3 different priors

    1. Sparse Adjacency Matrix (Feng et al., 2020)
    2. Gaussian Adjacency Matrix & Normalization as per (Kipf & Welling et al., ICLR 2017)
    3. Trainable Adjacency Matrix (Predict Edges)

## Usage

    1. python graph_neural_network.py --model fc
    2. python graph_neural_network.py --model conv
    3. python graph_neural_network.py --model gaussian_graph
    4. python graph_neural_network.py --model graph --pred_edge
    5. python graph_neural_network.py --model graph

## Visualize Filter

1. Sparse Filter
    ![Sparse](images\sparse_filter.gif)

2. Gaussian Filter
    ![Gaussian](images\gaussian_filter.gif)
