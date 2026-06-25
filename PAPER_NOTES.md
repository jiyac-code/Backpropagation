# Paper
Learning representations by back-propagating errors
By: David E. Rumelhart, Geoffrey E. Hintont & Ronald J. Williams

## Central Claim
The paper claims that for multi-layer neural networks can be trained effectively by the use of backpropagation. In backpropagation, the output errors are propagated backward through the network and the weights are adjusted using gradient descent.
Previously, it was difficult to train networks with hidden layers as there was no effective method to determine how the hidden weights should be updated. The backpropagation algorithm is a solution for this problem. It uses chain rule from calculus to update weights

## Core Algorithm and Architecture
The core algorithm is backpropagation. The network performs forward pass, calculates error and then propogates the error backwards to calculate gradients. These gradients are used to update weights using gradient descent with momentum.

Error(SSE): 

Back Propagation:  
Gradient calculation:  


Momentum Weight Update:  


I implemented a symmetry-detection network as described in the paper. It checks whether a 6 bit binary number is symmetric or not. The network architecture is:  
- 6 inputs
- 2 hidden neurons
- 1 output
- sigmoid activation function

## Dataset, Metrics, Baseline
Dataset:  
- All 64 possible combinations of 6 bit binary numbers  
- If symmetric, it outputs 1 else 0  

Metrics:
- Sum of Squared Error (SSE): The paper uses this metric. In my implementation, the SSE reduced from 9.9289 at pass 1 to 0.0051 at pass 14250. This shows that backpropagation worked in updating and improving the neural network.

Baseline:  
Earlier single-layer networks could not solve many non-linear problems. The paper shows that multilayer networks trained with backpropagation can successfully learn such tasks.

This implementation supports this paper's claims.


