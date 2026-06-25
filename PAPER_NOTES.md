# Paper
Learning representations by back-propagating errors
By: David E. Rumelhart, Geoffrey E. Hintont & Ronald J. Williams

## Central Claim
The paper claims that for multi-layer neural networks can be trained effectively by the use of backpropagation. In backpropagation, the output errors are propagated backward through the network and the weights are adjusted using gradient descent.
Previously, it was difficult to train networks with hidden layers as there was no effective method to determine how the hidden weights should be updated. The backpropagation algorithm is a solution for this problem. It uses chain rule from calculus to update weights

## Core Algorithm and Architecture
The core algorithm is backpropagation. The network performs forward pass, calculates error and then propogates the error backwards to calculate gradients. These gradients are used to update weights using gradient descent with momentum.

Error(SSE): 
<img width="212" height="52" alt="image" src="https://github.com/user-attachments/assets/b7581637-07f1-4cb1-8117-aedb2f303f2d" />

Back Propagation  
Gradient calculation:   
<img width="275" height="42" alt="image-1" src="https://github.com/user-attachments/assets/979e0ee7-8d89-4d35-a174-40a862879ef1" />
<img width="240" height="57" alt="image-2" src="https://github.com/user-attachments/assets/8871b59b-973d-4957-81b1-0db8f1e1f473" />

Momentum Weight Update:  
<img width="337" height="46" alt="image-3" src="https://github.com/user-attachments/assets/34ce53a9-3a79-407c-a93d-91c7286023da" />


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


