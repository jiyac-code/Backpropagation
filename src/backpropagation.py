import torch

final_sse = 0

# all 64 6-bit binary numbers
binary_vectors = []

for i in range(64):
    binary_string = f"{i:06b}"          
    binary_list = [int(x) for x in binary_string]
    binary_vectors.append(binary_list)

inputs = torch.tensor(binary_vectors, dtype=torch.float32)          #matrix of 64 rows and 6 columns


# if symmetrical, return 1 else 0
target_values = []

for row in inputs:
    left_half = row[:3]
    right_half = row[3:]

    reversed_right_half = torch.flip(right_half, dims=[0])          #flip the second half

    if torch.equal(left_half, reversed_right_half):                 #compare both halves and return 1 if symmetric
        target_values.append(1.0)
    else:
        target_values.append(0.0)

targets = torch.tensor(target_values,dtype=torch.float32).unsqueeze(1)  #kind of like a matrix with 1 column

torch.manual_seed(42)  #for reproducibility

#used this range so that sigmoid doesnt become saturated by very high or very low values
W1 = torch.empty(6, 2).uniform_(-0.3, 0.3)  #the 2 hidden weight
B1 = torch.empty(1, 2).uniform_(-0.3, 0.3)  #the 2 hidden biases (1 for each neuron)
W2 = torch.empty(2, 1).uniform_(-0.3, 0.3)  #the output weights
B2 = torch.empty(1, 1).uniform_(-0.3, 0.3)  #the output bias

#creating momentum vectors (to store previous updates)
velo_W1, velo_B1 = torch.zeros_like(W1), torch.zeros_like(B1)
velo_W2, velo_B2 = torch.zeros_like(W2), torch.zeros_like(B2)

#hyperparameters
eta = 0.01          #learning rate
alpha = 0.9         #momentum coeff
sweeps = 14250      #training passes

print(f"Total training passes: {sweeps}")

#Training
for sweep in range(1, sweeps + 1):

    #this is forward pass
    Z1 = torch.matmul(inputs, W1) + B1      #z=xw + b
    A1 = 1.0/(1.0+torch.exp(-Z1))       #sigmoid function: 1/(1 + e^-z)
    Z2 = torch.matmul(A1, W2) + B2
    A2 = 1.0/(1.0+torch.exp(-Z2))       #output from neuron
    
    #total sum of squared errors
    sse = 0.5*torch.sum((A2-targets)**2).item() #formula given in paper
    
    #this is backward pass (back propogation)
    dZ2 = (A2-targets) * (A2*(1.0-A2))    #formula 5 from the research paper
    

    #vectorized implementation of the equations in paper
    dZ1 = torch.matmul(dZ2, W2.t()) * (A1*(1.0-A1))
    
    #gradient calculation
    grad_W2 = torch.matmul(A1.t(), dZ2)
    grad_B2 = torch.sum(dZ2, dim=0, keepdim=True)
    grad_W1 = torch.matmul(inputs.t(), dZ1)
    grad_B1 = torch.sum(dZ1, dim=0, keepdim=True)
    
    #momentum updation using formula 9
    velo_W2 = -eta * grad_W2 + alpha * velo_W2
    velo_B2 = -eta * grad_B2 + alpha * velo_B2
    velo_W1 = -eta * grad_W1 + alpha * velo_W1
    velo_B1 = -eta * grad_B1 + alpha * velo_B1
    
    #apply modifications
    W2 += velo_W2
    B2 += velo_B2
    W1 += velo_W1
    B1 += velo_B1

    if sweep==1 or sweep % 1000 == 0 or sweep == sweeps:
        print(f"Training Pass {sweep:4d} | Total SSE: {sse:.4f}")
    
    if sweep==14250:
        final_sse=sse

#results
print("\n")
print("Verification")
print(f"The final SSE is: {final_sse:.4f}")
