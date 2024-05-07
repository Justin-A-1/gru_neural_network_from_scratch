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
        self.bias_r = np.zeros(hidden_size)
        self.bias_z = np.zeros(hidden_size)
        self.bias_h = np.zeros(hidden_size)
    
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
        r = sigmoid.function(np.matmul(self.W_r, x) + np.matmul(self.U_r, h_prev) + self.bias_r)
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
        z = sigmoid.function(np.matmul(self.W_z, x) + np.matmul(self.U_z, h_prev) + self.bias_z)
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
        h_candidate = tanh.function(np.matmul(self.W, x) + r * np.matmul(self.U, h_prev) + self.bias_h)
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
        r = self.reset_gate(x, h_prev)
        z = self.update_gate(x, h_prev)
        h_candidate = self.candidate_hidden_state(x, h_prev, r)
        h_next = z * h_prev + (1 - z) * h_candidate
        return h_next, r, z, h_candidate
    
    def backward_candidate(self, x, r, z, h_prev, h_candidate, h_next, h_actual):
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
        d_h_candidate_d_tanh_input = tanh.derivative(np.matmul(self.W, x) + r * np.matmul(self.U, h_prev) + self.bias_h) #d(tanh)

        # combine prior 3 terms:
        d_loss_d_tanh_input = d_loss_d_h_next * d_h_next_d_h_candidate * d_h_candidate_d_tanh_input

        # for W: d_loss/dW = d_loss/d_tanh_input * transpose(x)
        d_loss_dW = np.outer(d_loss_d_tanh_input, x)

        # for U: d_loss/dU = d_loss/d_tanh_input * transpose(r*h_prev)
        d_loss_dU = np.outer(d_loss_d_tanh_input, (r*h_prev))

        # for bias_h: derivative of bias_h term with respect to itself is 1, 
        # so the chain rule does not require another multiplication here:
        d_loss_d_bias_h = d_loss_d_tanh_input * 1
        
        return d_loss_dW, d_loss_dU, d_loss_d_bias_h
         
    
    def backward_update(self, x, r, z, h_prev, h_candidate, h_next, h_actual):
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
        d_z_d_sigmoid_input = sigmoid.derivative(np.matmul(self.W_z, x) + np.matmul(self.U_z, h_prev) + self.bias_z)

        # combine the prior 3 terms:
        d_loss_d_sigmoid_input = d_loss_d_h_next * d_h_next_d_z * d_z_d_sigmoid_input

        # for W_z: d_loss/dW_z = d_loss/d_sigmoid_input * transpose(x)
        d_loss_dW_z = np.outer(d_loss_d_sigmoid_input, x)

        # for U_z: d_loss/dU_z = d_loss/d_sigmoid_input * transpose(h_prev)
        d_loss_dU_z = np.outer(d_loss_d_sigmoid_input, h_prev)

        # for bias_z: derviation of bias_z term with respect to itself is 1,
        # so the chain rule does not require another multiplication here:
        d_loss_d_bias_z = d_loss_d_sigmoid_input * 1

        return d_loss_dW_z, d_loss_dU_z, d_loss_d_bias_z

    def backward_reset(self, x, r, z, h_prev, h_candidate, h_next, h_actual):
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
        d_h_candidate_d_tanh = tanh.derivative(np.matmul(self.W, x) + r * np.matmul(self.U, h_prev) + self.bias_h) #d(tanh)
        d_tanh_d_reset_gate = self.U*h_prev #d(U*h_prev)
        d_reset_gate_d_sigmoid_input = sigmoid.derivative(np.matmul(self.W_r, x) + np.matmul(self.U_r, h_prev) + self.bias_r)

        # combine prior 5 terms:
        d_loss_d_sigmoid_input = d_loss_d_h_next * d_h_next_d_h_candidate * d_h_candidate_d_tanh * d_tanh_d_reset_gate * d_reset_gate_d_sigmoid_input

        # for W_r: d_loss/dW_r = d_loss/d_sigmoid_input * transpose(x)
        d_loss_dW_r = np.outer(d_loss_d_sigmoid_input, x)

        # for U_r: d_loss/dU_r = d_loss/d_sigmoid_input * transpose(h_prev)
        d_loss_dU_r = np.outer(d_loss_d_sigmoid_input, h_prev)

        # for bias_r: derviation of bias_r term with respect to itself is 1,
        # so the chain rule does not require another multiplication here:
        d_loss_d_bias_r = d_loss_d_sigmoid_input * 1

        return d_loss_dW_r, d_loss_dU_r, d_loss_d_bias_r

    def backward_h_prev(self, x, r, z, h_prev, h_candidate, h_next, h_actual):
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

        d_z_d_sigmoid_input = sigmoid.derivative(np.matmul(self.W_z, x) + np.matmul(self.U_z, h_prev) + self.bias_z)
        d_sigmoid_input_d_h_prev = self.U_z
        d_z_d_h_prev = d_z_d_sigmoid_input * d_sigmoid_input_d_h_prev

        d_h_candidate_d_tanh = tanh.derivative(np.matmul(self.W, x) + r * np.matmul(self.U, h_prev) + self.bias_h) #d(tanh)
        sigmoid_derivative = sigmoid.derivative(np.matmul(self.W_r, x) + np.matmul(self.U_r, h_prev) + self.bias_r)
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
    
    # def backward_combined(self, x, r, z, h_prev, h_candidate, h_next, h_actual):

def main():
    # Initialize the GRU cell
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
    print("Candidate Hidden State (h_candidate):", h_candidate)

if __name__ == '__main__':
    main()
