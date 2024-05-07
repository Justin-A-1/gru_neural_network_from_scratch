import numpy as np

# see notes under class gru_cell for some notes about how gru's work

class sigmoid:
    @staticmethod
    def function(x):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def derivative(x):
        """Compute the derivative of the sigmoid function."""
        sigmoid_output = sigmoid.function(x)
        return sigmoid_output * (1 - sigmoid_output)

class tanh:
    @staticmethod
    def function(x):
        """Compute the hyperbolic tangent function."""
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def derivative(x):
        """Compute the derivative of the tanh function."""
        tanh_output = tanh.function(x)
        return 1 - tanh_output**2
    
class mean_square_error:
    @staticmethod
    def function(pred, act):
        L = (1/2)*(act - pred)**2
        L = L.sum()
        return L
    
    @staticmethod
    def derivative(pred, act):
        der = -(act - pred)
        return der

def orthogonal_init(shape):
    """Generate a random orthogonal matrix."""
    a = np.random.randn(*shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    return u if u.shape == shape else v

class gru_cell:
    '''
    - when making this code, I learned about gru's here, which you may find helpful:
    https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be
    - when making the back propagation code, i found these slides on neural networks to be very helpful

    HOW DO GRU's WORK?
    - gru aka gated recurrent unit aka gated recurrent neural network
    - each gru cell corresponds roughly to a single hidden layer of a regular, forward neural network
    - gru_cell mitigates vanishing gradient problems of RNN in a cost-efficient manner by
    effectively having 3 hidden layer componenets for each layer; as in, 3 sets of nodes, each with their
    own weights and biases
    - the three hidden layer components are:
    --- 1. the candidate layer - this layer correspondes to the layer you know and love from regular, forward neural networks
    --- 2. the reset layer - this layer gives the model chance to ask: "what do I want to eliminate/scale down 
    from previous hidden state info?"
    --- 3. the update layer - this layer gives the model the chance to ask: "how much do I want the 
    next hidden state to be based on the candidate hidden state vs the parly-reset previous hidden state?"

    --- also note: each of these layers has input from ordinary input as well as previous hidden state
    --- these two input components result in 2*3 times as many sets of weights as regular, forward neural networks

    - there is substantial overlap between the role of 2 and 3, but don't worry about that! gpu's work!
    '''
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights orthogonally for hidden weight matrices
        self.U_r = orthogonal_init((hidden_size, hidden_size))
        self.U_z = orthogonal_init((hidden_size, hidden_size))
        self.U = orthogonal_init((hidden_size, hidden_size))

        # Initialize input weight matrices randomly
        self.W_r = np.random.randn(hidden_size, input_size)
        self.W_z = np.random.randn(hidden_size, input_size)
        self.W = np.random.randn(hidden_size, input_size)

        # Bias terms, initialized to zero
        self.bias_r = np.zeros((hidden_size, 1))
        self.bias_z = np.zeros((hidden_size, 1))
        self.bias_h = np.zeros((hidden_size, 1))
    
    def reset_gate(self, x, h_prev):
        """
        Calculate the reset gate using the current input and previous hidden state.
    
        The reset gate determines how much of the previous hidden state will be erased before being 
        considered to be added to the next hiddent state. It uses the sigmoid function 
        to ensure the output is between 0 and 1, which scales the previous hidden state accordingly.

        Args:
            x (np.array): The input vector at the current time step.
            h_prev (np.array): The hidden state vector from the previous time step.

        Returns:
            r (np.array): The reset gate output
        """
        #print("CALCULATING r...")

        '''print(f"x shape: {x.shape} (expected: ({self.input_size}, 1))")
        print(f"h_prev shape: {h_prev.shape} (expected: ({self.hidden_size}, 1))")
        print(f"W_r shape: {self.W_r.shape} (expected: ({self.hidden_size}, {self.input_size}))")
        print(f"U_r shape: {self.U_r.shape} (expected: ({self.hidden_size}, {self.hidden_size}))")
        print(f"bias_r shape: {self.bias_r.shape} (expected: ({self.hidden_size},))")'''

        r = sigmoid.function(np.matmul(self.W_r, x) + np.matmul(self.U_r, h_prev) + self.bias_r)

        '''print(f"r shape: {r.shape} (expected: ({self.hidden_size}, 1))")'''
        return r

    def update_gate(self, x, h_prev):
        """
        Calculate the update gate using the current input and previous hidden state.
        
        The update gate determines what percent of the updated hidden state will come from the 
        curent state vs the previous state. It uses the sigmoid function to create a gate 
        value between 0 and 1.

        Args:
            x (np.array): The input vector at the current time step.
            h_prev (np.array): The hidden state vector from the previous time step.

        Returns:
            z (np.array): The update gate output
        """
        '''print("CALCULATING z...")
        print(f"x shape: {x.shape} (expected: ({self.input_size}, 1))")
        print(f"h_prev shape: {h_prev.shape} (expected: ({self.hidden_size}, 1))")
        print(f"W_z shape: {self.W_z.shape} (expected: ({self.hidden_size}, {self.input_size}))")
        print(f"U_z shape: {self.U_z.shape} (expected: ({self.hidden_size}, {self.hidden_size}))")
        print(f"bias_z shape: {self.bias_z.shape} (expected: ({self.hidden_size},))")'''

        z = sigmoid.function(np.matmul(self.W_z, x) + np.matmul(self.U_z, h_prev) + self.bias_z)

        '''print(f"z shape: {z.shape} (expected: ({self.hidden_size}, 1))")'''
        
        return z

    def candidate_hidden_state(self, x, h_prev, r):
        """
        Calculate the candidate hidden state using the current input, previous hidden state, and the reset gate output.
        
        This method computes what the new hidden state could be if it were to be completely replaced. This potential
        new state is modulated by the reset gate, which scales the previous hidden state before combining it with
        the current input. The tanh function ensures that the output is between -1 and 1, which helps regulate the network's activations.

        Args:
            x (np.array): The input vector at the current time step.
            h_prev (np.array): The hidden state vector from the previous time step
            r (np.array): The output from the reset gate

        Returns:
            h_candidate (np.array): The candidate hidden state, which is a combination of the current input and
                                the scaled previous hidden state, passed through a tanh activation function.
        """
        #print("CALCULATING h_candidate...")
        '''print(f"x shape: {x.shape} (expected: ({self.input_size}, 1))")
        print(f"h_prev shape: {h_prev.shape} (expected: ({self.hidden_size}, 1))")
        print(f"r shape: {r.shape} (expected: ({self.hidden_size}, 1))")
        print(f"W shape: {self.W.shape} (expected: ({self.hidden_size}, {self.input_size}))")
        print(f"U shape: {self.U.shape} (expected: ({self.hidden_size}, {self.hidden_size}))")
        print(f"bias_h shape: {self.bias_h.shape} (expected: ({self.hidden_size},))")'''

        h_candidate = tanh.function(np.matmul(self.W, x) + r * np.matmul(self.U, h_prev) + self.bias_h)

        '''print(f"h_candidate shape: {h_candidate.shape} (expected: ({self.hidden_size}, 1))")'''
        return h_candidate

    def forward(self, x, h_prev):
        """
        Perform a forward pass of the GRU cell using the current input x and previous hidden state h_prev.
        
        Args:
            x (np.array): The input vector at the current time step.
            h_prev (np.array): The hidden state vector from the previous time step.

        Returns:
            h_next (np.array): The next hidden state vector.
            r (np.array): The reset gate's output.
            z (np.array): The update gate's output.
            h_candidate (np.array): The candidate hidden state, commonly referred to as h_tilde in documentation.
        """
        #print(f"BEGINNING FORWARD PASS...")
        r = self.reset_gate(x, h_prev)
        #print(f"The shape of r is {r.shape}")
        z = self.update_gate(x, h_prev)
        #print(f"The shape of z is {z.shape}")
        h_candidate = self.candidate_hidden_state(x, h_prev, r)
        #print(f"The shape of h_candidate is {h_candidate.shape}")
        h_next = z * h_prev + (1 - z) * h_candidate
        #print(f"The shape of h_next is {h_next.shape}")
        return h_next, r, z, h_candidate
    
    def backward_candidate(self, step_size, x, r, z, h_prev, h_candidate, h_next, h_actual):
        # back propagation for candidate associated values: W, U, and bias_h

        # derivatives for these lines:
        # loss calculation: L = (1/2)*(act - pred)**2
        # h_next prediction: h_next = z * h_prev + (1 - z) * h_candidate
        # h_candidate calculation: h_candidate = tanh.function(np.matmul(self.W, x) + r * np.matmul(self.U, h_prev) + self.bias_h)

        # note: each derivative is the same - until you get to the tanh inputs
        # d_loss/d_tanh_input = dL/d_h_next * d_h_next/d_h_candidate * d_h_candidate/d_tanh_input
        # d_loss/d_tanh_input = d(MSE) * d(update) * d(tanh)
        d_loss_d_h_next = -(h_actual - h_next) #d(Mean Square Error)/d_h_next
        d_h_next_d_h_candidate = (1-z) #d(update)
        tanh_input = np.matmul(self.W, x) + r * np.matmul(self.U, h_prev) + self.bias_h
        d_h_candidate_d_tanh_input = tanh.derivative(tanh_input) #d(tanh)

        #print(f"The shape of d_loss_d_h_next is {d_loss_d_h_next.shape}")
        #print(f"The shape of d_h_next_d_h_candidate is {d_h_next_d_h_candidate.shape}")
        #print(f"The shape of d_h_candidate_d_tanh_input is {d_h_candidate_d_tanh_input.shape}")

        # combine prior 3 terms:
        d_loss_d_tanh_input = d_loss_d_h_next * d_h_next_d_h_candidate * d_h_candidate_d_tanh_input

        #print(f"The shape of d_loss_d_tanh_input is {d_loss_d_tanh_input.shape}")

        # for W: d_loss/dW = d_loss/d_tanh_input * transpose(x)
        d_loss_dW = np.outer(d_loss_d_tanh_input, x)

        #print(f"The shape of d_loss_dW is {d_loss_dW.shape}")

        # for U: d_loss/dU = d_loss/d_tanh_input * transpose(r*h_prev)
        d_loss_dU = np.outer(d_loss_d_tanh_input, (r*h_prev))

        #print(f"The shape of d_loss_dU is {d_loss_dU.shape}")

        # for bias_h: derivative of bias_h term with respect to itself is 1, 
        # so the chain rule does not require another multiplication here:
        d_loss_d_bias_h = d_loss_d_tanh_input * 1
        
        #print(f"The shape of d_loss_d_bias_h is {d_loss_d_bias_h.shape}")

        # update parameters
        self.W = self.W - step_size*d_loss_dW
        self.U = self.U - step_size*d_loss_dU
        self.bias_h = self.bias_h - step_size*d_loss_d_bias_h

        # return d_loss_dW, d_loss_dU, d_loss_d_bias_h
         
    def backward_update(self, step_size, x, r, z, h_prev, h_candidate, h_next, h_actual):
        # back propagation for the update associated values: W_z, U_z, and bias_z

        # derivatives for these lines:
        # loss calculation: L = (1/2)*(act - pred)**2
        # h_next prediction: h_next = z * h_prev + (1 - z) * h_candidate
        # update gate calculation: z = sigmoid.function(np.matmul(self.W_z, x) + np.matmul(self.U_z, h_prev) + self.bias_z)

        # NOTE: each derivative is the same - until you get to the sigmoid inputs
        # d_loss/d_sigmoid_input = dL/d_h_next * d_h_next/d_z * d_z/d_sigmoid_input
        # d_loss/d_sigmoid_input = d(MSE) * d(update) * d(sigmoid)
        d_loss_d_h_next = -(h_actual - h_next) #d(Mean Square Error)/d_h_next
        d_h_next_d_z = h_prev - h_candidate #d(update)
        sigmoid_input = np.matmul(self.W_z, x) + np.matmul(self.U_z, h_prev) + self.bias_z
        d_z_d_sigmoid_input = sigmoid.derivative(sigmoid_input)

        #print(f"The shape of d_loss_d_h_next is {d_loss_d_h_next.shape}")
        #print(f"The shape of d_h_next_d_z is {d_h_next_d_z.shape}")
        #print(f"The shape of d_z_d_sigmoid_input is {d_z_d_sigmoid_input.shape}")

        # combine the prior 3 terms:
        d_loss_d_sigmoid_input = d_loss_d_h_next * d_h_next_d_z * d_z_d_sigmoid_input

        #print(f"The shape of d_loss_d_sigmoid_input is {d_loss_d_sigmoid_input.shape}")

        # for W_z: d_loss/dW_z = d_loss/d_sigmoid_input * transpose(x)
        d_loss_dW_z = np.outer(d_loss_d_sigmoid_input, x)

        #print(f"The shape of d_loss_dW_z is {d_loss_dW_z.shape}")

        # for U_z: d_loss/dU_z = d_loss/d_sigmoid_input * transpose(h_prev)
        d_loss_dU_z = np.outer(d_loss_d_sigmoid_input, h_prev)

        #print(f"The shape of d_loss_dU_z is {d_loss_dU_z.shape}")

        # for bias_z: derviation of bias_z term with respect to itself is 1,
        # so the chain rule does not require another multiplication here:
        d_loss_d_bias_z = d_loss_d_sigmoid_input * 1

        #print(f"The shape of d_loss_d_bias_z is {d_loss_d_bias_z.shape}")

        # update parameters
        self.W_z = self.W_z - step_size*d_loss_dW_z
        self.U_z = self.U_z - step_size*d_loss_dU_z
        self.bias_z = self.bias_z - step_size*d_loss_d_bias_z

        #return d_loss_dW_z, d_loss_dU_z, d_loss_d_bias_z

    def backward_reset(self, step_size, x, r, z, h_prev, h_candidate, h_next, h_actual):
        # back propagation for reset associated values: W_r, U_r, and bias_r

        # derivatives for these lines:
        # loss calculation: L = (1/2)*(act - pred)**2
        # h_next prediction: h_next = z * h_prev + (1 - z) * h_candidate
        # h_candidate calculation: h_candidate = tanh.function(np.matmul(self.W, x) + r * np.matmul(self.U, h_prev) + self.bias_h)
        # reset gate calculation: r = sigmoid.function(np.matmul(self.W_r, x) + np.matmul(self.U_r, h_prev) + self.bias_r)

        # NOTE: each derivative is the same - until you get to the sigmoid inputs
        # d_loss/d_sigmoid_input = dL/d_h_next * d_h_next/d_h_candidate * d_h_candidate/d_tanh * d_tanh/d_reset_gate * d_reset_gate/d_sigmoid_input
        # d_loss/d_sigmoid_input = d(MSE) * d(update) * d(tanh) * d(U*h_prev) * d(sigmoid)

        d_loss_d_h_next = -(h_actual - h_next) #d(Mean Square Error)/d_h_next
        d_h_next_d_h_candidate = (1-z) #d(update)
        tanh_input = np.matmul(self.W, x) + r * np.matmul(self.U, h_prev) + self.bias_h
        d_h_candidate_d_tanh = tanh.derivative(tanh_input) #d(tanh)
        d_tanh_d_reset_gate = np.matmul(self.U, h_prev) #previously had self.U*h_prev #d(U*h_prev)
        sigmoid_input = np.matmul(self.W_r, x) + np.matmul(self.U_r, h_prev) + self.bias_r
        d_reset_gate_d_sigmoid_input = sigmoid.derivative(sigmoid_input)

        '''print(f"d_loss_d_h_next shape: {d_loss_d_h_next.shape} (expected size: (hidden_size, 1) aka ({self.hidden_size}, 1))")
        print(f"d_h_next_d_h_candidate shape: {d_h_next_d_h_candidate.shape} (expected size: (hidden_size, 1) aka ({self.hidden_size}, 1))")
        print(f"d_h_candidate_d_tanh shape: {d_h_candidate_d_tanh.shape} (expected size: (hidden_size, 1) aka ({self.hidden_size}, 1))")
        print(f"self.U shape: {self.U.shape} (expected size: (hidden_size, 1) aka ({self.hidden_size}, {self.hidden_size}))")
        print(f"h_prev shape: {h_prev.shape} (expected size: (hidden_size, 1) aka ({self.hidden_size}, 1))")
        print(f"d_tanh_d_reset_gate shape: {d_tanh_d_reset_gate.shape} (expected size: (hidden_size, 1) aka ({self.hidden_size}, 1))")
        print(f"d_reset_gate_d_sigmoid_input shape: {d_reset_gate_d_sigmoid_input.shape} (expected size: (hidden_size, 1) aka ({self.hidden_size}, 1))")'''

        # combine prior 5 terms:
        d_loss_d_sigmoid_input = d_loss_d_h_next * d_h_next_d_h_candidate * d_h_candidate_d_tanh * d_tanh_d_reset_gate * d_reset_gate_d_sigmoid_input
        #print(f"d_loss_d_sigmoid_input shape: {d_loss_d_sigmoid_input.shape} (expected size: (hidden_size, 1) aka ({self.hidden_size}, 1))")

        # for W_r: d_loss/dW_r = d_loss/d_sigmoid_input * transpose(x)
        d_loss_dW_r = np.outer(d_loss_d_sigmoid_input, x)
        #print(f"d_loss_dW_r shape: {d_loss_dW_r.shape} (expected size: (hidden_size, input_size) aka ({self.hidden_size}, {self.input_size}))")

        # for U_r: d_loss/dU_r = d_loss/d_sigmoid_input * transpose(h_prev)
        d_loss_dU_r = np.outer(d_loss_d_sigmoid_input, h_prev)
        #print(f"d_loss_dU_r shape: {d_loss_dU_r.shape} (expected size: (hidden_size, hidden_size) aka ({self.hidden_size}, {self.hidden_size}))")

        # for bias_r: derviation of bias_r term with respect to itself is 1,
        # so the chain rule does not require another multiplication here:
        d_loss_d_bias_r = d_loss_d_sigmoid_input * 1
        #print(f"d_loss_d_bias_r shape: {d_loss_d_bias_r.shape} (expected size: (hidden_size, 1) aka ({self.hidden_size}, 1))")

        # update values
        self.W_r = self.W_r - step_size*d_loss_dW_r
        #print(f"Updated self.W_r shape: {self.W_r.shape} (expected size: (hidden_size, input_size) aka ({self.hidden_size}, {self.input_size}))")
        self.U_r = self.U_r - step_size*d_loss_dU_r
        #print(f"Updated self.U_r shape: {self.U_r.shape} (expected size: (hidden_size, hidden_size) aka ({self.hidden_size}, {self.hidden_size}))")
        self.bias_r = self.bias_r - step_size*d_loss_d_bias_r
        #print(f"Updated self.bias_r shape: {self.bias_r.shape} (expected size: (hidden_size, 1) aka ({self.hidden_size}, 1))")

        #return d_loss_dW_r, d_loss_dU_r, d_loss_d_bias_r

    def backward_h_prev(self, step_size, x, r, z, h_prev, h_candidate, h_next, h_actual):
        # back propagation for h_prev
        # this is essential for the GRU to have memory of previous data items
        # h_prev touches every part of the process, so its derivative is a lot!

        # derivatives for these lines:
        # loss calculation: L = (1/2)*(act - pred)**2
        # h_next prediction: h_next = z * h_prev + (1 - z) * h_candidate
        # h_candidate calculation: h_candidate = tanh.function(np.matmul(self.W, x) + r * np.matmul(self.U, h_prev) + self.bias_h)
        # reset gate calculation: r = sigmoid.function(np.matmul(self.W_r, x) + np.matmul(self.U_r, h_prev) + self.bias_r)
        # update gate calculation: z = sigmoid.function(np.matmul(self.W_z, x) + np.matmul(self.U_z, h_prev) + self.bias_z)

        # d_loss/d_h_prev = dL/d_h_next * [z*(d_h_prev/d_h_prev=1) + h_prev*(d_z/d_h_prev) + (1-z)*d_h_candidate/d_h_prev... 
        # ... + h_candidate*(-d_z/d_h_prev)]      
        # d_z/d_h_prev = d_z/d_sigmoid_input * d_sigmoid_input/d_h_prev   
        # d_h_candidate/d_h_prev = d_h_candidate/d_tanh_input * ...
        # ... [ (reset_gate*U*1=d_tanh_input/d_h_prev) + (U*h_prev=d_tanh_input/d_reset_gate)*d_reset_gate/d_sigmoid_input*d_sigmoid/d_h_prev]    
        d_loss_d_h_next = -(h_actual - h_next) #d(Mean Square Error)/d_h_next

        sigmoid_input_z = np.matmul(self.W_z, x) + np.matmul(self.U_z, h_prev) + self.bias_z
        d_z_d_sigmoid_input = sigmoid.derivative(sigmoid_input_z)
        d_sigmoid_input_d_h_prev = self.U_z
        d_z_d_h_prev = d_z_d_sigmoid_input * d_sigmoid_input_d_h_prev

        tanh_input = np.matmul(self.W, x) + r * np.matmul(self.U, h_prev) + self.bias_h
        d_h_candidate_d_tanh = tanh.derivative(tanh_input) #d(tanh)
        sigmoid_input_r = np.matmul(self.W_r, x) + np.matmul(self.U_r, h_prev) + self.bias_r
        sigmoid_derivative = sigmoid.derivative(sigmoid_input_r)
        reset_gate_deriv_side = np.matmul(sigmoid_derivative*self.U_r, np.matmul(self.U, h_prev))
        h_prev_derive_side = (r*self.U)*1

        d_h_candidate_d_h_prev = d_h_candidate_d_tanh*(reset_gate_deriv_side + h_prev_derive_side)

        d_h_next_d_h_prev = z*1 + h_prev*d_z_d_h_prev + h_candidate*(-d_z_d_h_prev) + (1-z)*d_h_candidate_d_h_prev
        d_loss_d_h_prev = d_loss_d_h_next * d_h_next_d_h_prev
        return d_loss_d_h_prev

    # NOTE: if you wish to do multilayer GRU, you would also have to add something like:
    # def backward_input(self, x, r, z, h_prev, h_candidate, h_next, h_actual):
    # this is because a multilayer GRU uses output from one layer to act as input to 
    # the next (separate from time step iteration)
    
    def backward_combined(self, step_size, x, r, z, h_prev, h_candidate, h_next, h_actual):
        self.backward_candidate(step_size, x, r, z, h_prev, h_candidate, h_next, h_actual)
        self.backward_update(step_size, x, r, z, h_prev, h_candidate, h_next, h_actual)
        self.backward_reset(step_size, x, r, z, h_prev, h_candidate, h_next, h_actual)
        d_loss_d_h_prev = self.backward_h_prev(step_size, x, r, z, h_prev, h_candidate, h_next, h_actual)

        # only d_loss_d_h_prev is returned. All other gradients are immediately used to update values in the current
        # gru cell, but d_loss_d_h_prev is used as error for the previous cell
        return d_loss_d_h_prev

def generate_series(length):
    # Generate a simple linearly increasing series
    return np.arange(length)

def train_gru(gru_cell, series, epochs, learning_rate):
    # Assume series is already reshaped appropriately for the input to the GRU
    h_prev = np.zeros((gru_cell.hidden_size, 1))  # Initial hidden state as a matrix of 1 column
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for t in range(len(series) - 1):
            x = series[t:t+1]  # Current input slice
            h_actual = series[t+1:t+2]  # Next input in the series as the actual output

            # Forward pass
            h_next, r, z, h_candidate = gru_cell.forward(x, h_prev)
            loss = mean_square_error.function(h_next, h_actual)
            total_loss += loss

            # Backward pass
            d_loss_d_h_next = mean_square_error.derivative(h_next, h_actual)
            gru_cell.backward_combined(learning_rate, x, r, z, h_prev, h_candidate, h_next, h_actual)

            # Update previous hidden state
            h_prev = h_next

        # Store or print the loss after each epoch
        losses.append(total_loss)
        print(f'Epoch {epoch+1}, Total Loss: {total_loss}')
    
    return losses

def main():
    # Main execution setup
    input_size = 1  # As series is 1-dimensional
    hidden_size = 5
    gru_cell_1 = gru_cell(input_size, hidden_size)

    # Generate and train on a simple series
    series_length = 100
    series = generate_series(series_length)
    series = series.reshape(-1, 1)  # Reshape series to fit input size expected by GRU

    # Train the GRU
    epochs = 10
    learning_rate = 0.1
    losses = train_gru(gru_cell_1, series, epochs, learning_rate)

'''    # Initialize the GRU cell
    input_size = 3
    hidden_size = 5
    gru_cell_1 = gru_cell(input_size, hidden_size)

    # Define dummy input and previous hidden state
    x = np.random.randn(input_size)  # Random input vector
    h_prev = np.random.randn(hidden_size)  # Random previous hidden state vector

    # Perform a forward pass
    h_next, r, z, h_candidate = gru_cell_1.forward(x, h_prev)

    # Print the outputs to see what's happening
    print("Next Hidden State (h_next):", h_next)
    print("Reset Gate Output (r):", r)
    print("Update Gate Output (z):", z)
    print("Candidate Hidden State (h_candidate):", h_candidate)'''

if __name__ == '__main__':
    main()