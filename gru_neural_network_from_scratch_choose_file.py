import numpy as np
import pandas as pd

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
        #print(f"BEGINNING FORWARD PASS...")
        r = self.reset_gate(x, h_prev)
        z = self.update_gate(x, h_prev)
        h_candidate = self.candidate_hidden_state(x, h_prev, r)
        h_next = z * h_prev + (1 - z) * h_candidate
        return h_next, r, z, h_candidate
    
    def backward_candidate(self, step_size, x, r, z, h_prev, h_candidate, h_next, d_loss_d_h_next):
        # back propagation for candidate associated values: W, U, and bias_h

        # derivatives for these lines:
        # loss calculation: L = (1/2)*(act - pred)**2
        # h_next prediction: h_next = z * h_prev + (1 - z) * h_candidate
        # h_candidate calculation: h_candidate = tanh.function(np.matmul(self.W, x) + r * np.matmul(self.U, h_prev) + self.bias_h)

        # note: each derivative is the same - until you get to the tanh inputs
        # d_loss/d_tanh_input = dL/d_h_next * d_h_next/d_h_candidate * d_h_candidate/d_tanh_input
        # d_loss/d_tanh_input = d(MSE) * d(update) * d(tanh)

        #d_loss_d_h_next = -(h_actual - h_next) #d(Mean Square Error)/d_h_next
        d_h_next_d_h_candidate = (1-z) #d(update)
        tanh_input = np.matmul(self.W, x) + r * np.matmul(self.U, h_prev) + self.bias_h
        d_h_candidate_d_tanh_input = tanh.derivative(tanh_input) #d(tanh)

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
         
    def backward_update(self, step_size, x, r, z, h_prev, h_candidate, h_next, d_loss_d_h_next):
        # back propagation for the update associated values: W_z, U_z, and bias_z

        # derivatives for these lines:
        # loss calculation: L = (1/2)*(act - pred)**2
        # h_next prediction: h_next = z * h_prev + (1 - z) * h_candidate
        # update gate calculation: z = sigmoid.function(np.matmul(self.W_z, x) + np.matmul(self.U_z, h_prev) + self.bias_z)

        # NOTE: each derivative is the same - until you get to the sigmoid inputs
        # d_loss/d_sigmoid_input = dL/d_h_next * d_h_next/d_z * d_z/d_sigmoid_input
        # d_loss/d_sigmoid_input = d(MSE) * d(update) * d(sigmoid)

        #d_loss_d_h_next = -(h_actual - h_next) #d(Mean Square Error)/d_h_next
        d_h_next_d_z = h_prev - h_candidate #d(update)
        sigmoid_input = np.matmul(self.W_z, x) + np.matmul(self.U_z, h_prev) + self.bias_z
        d_z_d_sigmoid_input = sigmoid.derivative(sigmoid_input)

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

    def backward_reset(self, step_size, x, r, z, h_prev, h_candidate, h_next, d_loss_d_h_next):
        # back propagation for reset associated values: W_r, U_r, and bias_r

        # derivatives for these lines:
        # loss calculation: L = (1/2)*(act - pred)**2
        # h_next prediction: h_next = z * h_prev + (1 - z) * h_candidate
        # h_candidate calculation: h_candidate = tanh.function(np.matmul(self.W, x) + r * np.matmul(self.U, h_prev) + self.bias_h)
        # reset gate calculation: r = sigmoid.function(np.matmul(self.W_r, x) + np.matmul(self.U_r, h_prev) + self.bias_r)

        # NOTE: each derivative is the same - until you get to the sigmoid inputs
        # d_loss/d_sigmoid_input = dL/d_h_next * d_h_next/d_h_candidate * d_h_candidate/d_tanh * d_tanh/d_reset_gate * d_reset_gate/d_sigmoid_input
        # d_loss/d_sigmoid_input = d(MSE) * d(update) * d(tanh) * d(U*h_prev) * d(sigmoid)

        #d_loss_d_h_next = -(h_actual - h_next) #d(Mean Square Error)/d_h_next
        d_h_next_d_h_candidate = (1-z) #d(update)
        tanh_input = np.matmul(self.W, x) + r * np.matmul(self.U, h_prev) + self.bias_h
        d_h_candidate_d_tanh = tanh.derivative(tanh_input) #d(tanh)
        d_tanh_d_reset_gate = np.matmul(self.U, h_prev) #previously had self.U*h_prev #d(U*h_prev)
        sigmoid_input = np.matmul(self.W_r, x) + np.matmul(self.U_r, h_prev) + self.bias_r
        d_reset_gate_d_sigmoid_input = sigmoid.derivative(sigmoid_input)

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

    def backward_h_prev(self, step_size, x, r, z, h_prev, h_candidate, h_next, d_loss_d_h_next):
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
        # d_loss_d_h_next = -(h_actual - h_next) #d(Mean Square Error)/d_h_next

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

        d_h_next_d_h_prev = (
            z*1 #+ #+ 
            # other components that were not properly debugged:
            #h_prev*.1 #+ #*d_z_d_h_prev #h_prev*d_z_d_h_prev #+ 
            #h_candidate*.1 #+#*(-d_z_d_h_prev) #h_candidate*(-d_z_d_h_prev) #+ 
            #(1-z)*.1 #*d_h_candidate_d_h_prev #(1-z)*d_h_candidate_d_h_prev
        )
        d_loss_d_h_prev = d_loss_d_h_next * d_h_next_d_h_prev
        return d_loss_d_h_prev

    # NOTE: if you wish to do multilayer GRU, you would also have to add something like:
    # def backward_input(self, x, r, z, h_prev, h_candidate, h_next, d_loss_d_h_next):
    # this is because a multilayer GRU uses output from one layer to act as input to 
    # the next (separate from time step iteration)
    
    def backward_combined(self, step_size, x, r, z, h_prev, h_candidate, h_next, d_loss_d_h_next):
        d_loss_dW, d_loss_dU, d_loss_d_bias_h = self.backward_candidate(step_size, x, r, z, h_prev, h_candidate, h_next, d_loss_d_h_next)
        d_loss_dW_z, d_loss_dU_z, d_loss_d_bias_z = self.backward_update(step_size, x, r, z, h_prev, h_candidate, h_next, d_loss_d_h_next)
        d_loss_dW_r, d_loss_dU_r, d_loss_d_bias_r = self.backward_reset(step_size, x, r, z, h_prev, h_candidate, h_next, d_loss_d_h_next)
        d_loss_d_h_prev = self.backward_h_prev(step_size, x, r, z, h_prev, h_candidate, h_next, d_loss_d_h_next)

        backward_results_dict = {
            'd_loss_dW': d_loss_dW,
            'd_loss_dU': d_loss_dU,
            'd_loss_d_bias_h': d_loss_d_bias_h,
            'd_loss_dW_z': d_loss_dW_z,
            'd_loss_dU_z': d_loss_dU_z,
            'd_loss_d_bias_z': d_loss_d_bias_z,
            'd_loss_dW_r': d_loss_dW_r,
            'd_loss_dU_r': d_loss_dU_r,
            'd_loss_d_bias_r': d_loss_d_bias_r,
        }

        return d_loss_d_h_prev, backward_results_dict
    
    def update_weights(self, results_dict, learning_rate):
        # Update hidden candidate weights and biases
        self.W -= learning_rate * results_dict['d_loss_dW']
        self.U -= learning_rate * results_dict['d_loss_dU']
        self.bias_h -= learning_rate * results_dict['d_loss_d_bias_h']

        # Update update gate weights and biases
        self.W_z -= learning_rate * results_dict['d_loss_dW_z']
        self.U_z -= learning_rate * results_dict['d_loss_dU_z']
        self.bias_z -= learning_rate * results_dict['d_loss_d_bias_z']

        # Update reset gate weights and biases
        self.W_r -= learning_rate * results_dict['d_loss_dW_r']
        self.U_r -= learning_rate * results_dict['d_loss_dU_r']
        self.bias_r -= learning_rate * results_dict['d_loss_d_bias_r']

def generate_series(length):
    # Generate a simple linearly increasing series
    return np.arange(length)

def train_gru(gru_cell_1, series, epochs, learning_rate, sequence_length):
    # Assume series is already reshaped appropriately for the input to the GRU
    h_prev_carry_over = np.zeros((gru_cell_1.hidden_size, 1))  # Initial hidden state as a matrix of 1 column
    losses = []
    num_sequences = len(series)//sequence_length

    for epoch in range(epochs):
        total_epoch_loss = 0
        #steps_in_sequence = 0  # Counter to track steps within the sequence

        for seq_num in range(num_sequences):
            backward_results_dict_total = {
                'd_loss_dW': np.zeros_like(gru_cell_1.W),
                'd_loss_dU': np.zeros_like(gru_cell_1.U),
                'd_loss_d_bias_h': np.zeros_like(gru_cell_1.bias_h),
                'd_loss_dW_z': np.zeros_like(gru_cell_1.W_z),
                'd_loss_dU_z': np.zeros_like(gru_cell_1.U_z),
                'd_loss_d_bias_z': np.zeros_like(gru_cell_1.bias_z),
                'd_loss_dW_r': np.zeros_like(gru_cell_1.W_r),
                'd_loss_dU_r': np.zeros_like(gru_cell_1.U_r),
                'd_loss_d_bias_r': np.zeros_like(gru_cell_1.bias_r),
            }

            backward_results_dict_d_h_prev = {
                'd_loss_d_h_prev': np.zeros((sequence_length, gru_cell_1.hidden_size, 1))
            }

            forward_results_dict = {
                'h_prev': np.zeros((sequence_length, gru_cell_1.hidden_size, 1)),
                'h_next': np.zeros((sequence_length, gru_cell_1.hidden_size, 1)),
                'r': np.zeros((sequence_length, gru_cell_1.hidden_size, 1)),
                'z': np.zeros((sequence_length, gru_cell_1.hidden_size, 1)),
                'h_candidate': np.zeros((sequence_length, gru_cell_1.hidden_size, 1)),
                'loss': np.zeros((sequence_length, gru_cell_1.hidden_size, 1)),
                'd_loss_d_h_next': np.zeros((sequence_length, gru_cell_1.hidden_size, 1))
            }
            # do forward passes and store results
            for forward_iter in range(sequence_length):
                time_step = forward_iter + sequence_length*seq_num # current time step
                x = series[time_step:time_step+1]  # Current input slice
                h_actual = series[time_step+1:time_step+2]  # Next input in the series as the actual output

                if forward_iter == 0: #load in the last h_prev from the previous time step sequence
                    forward_results_dict['h_prev'][forward_iter] = h_prev_carry_over

                # execute forward pass
                h_next, r, z, h_candidate = gru_cell_1.forward(x, forward_results_dict['h_prev'][forward_iter])

                loss = 0.5 * np.mean((h_actual - h_next) ** 2)  # mean square error
                d_loss_d_h_next = -(h_actual - h_next)

                forward_results_dict['h_next'][forward_iter] = h_next
                forward_results_dict['r'][forward_iter] = r
                forward_results_dict['z'][forward_iter] = z
                forward_results_dict['h_candidate'][forward_iter] = h_candidate
                forward_results_dict['loss'][forward_iter] = loss
                forward_results_dict['d_loss_d_h_next'][forward_iter] = d_loss_d_h_next

                if forward_iter != (sequence_length - 1):
                    forward_results_dict['h_prev'][forward_iter+1] = h_next
                else:
                    h_prev_carry_over = h_next #store the last h_next to use when starting the next time step sequence
                
                total_epoch_loss += loss

            # do backward passes, accessing results from forward pass
            for backward_iter in range(sequence_length - 1, -1, -1):
                time_step = backward_iter + sequence_length*seq_num # current time step
                x = series[time_step:time_step+1]  # Current input slice

                # unless it's the last item in the iteration (the first item when going backwards):
                # add the d_loss_d_h_prev from the previous iteration
                if backward_iter != (sequence_length - 1):
                    forward_results_dict['d_loss_d_h_next'][backward_iter] = (
                        forward_results_dict['d_loss_d_h_next'][backward_iter] + 
                        backward_results_dict_d_h_prev['d_loss_d_h_prev'][backward_iter+1]
                    )

                r = forward_results_dict['r'][backward_iter]
                z = forward_results_dict['z'][backward_iter]
                h_prev = forward_results_dict['h_prev'][backward_iter]
                h_candidate = forward_results_dict['h_candidate'][backward_iter]
                h_next = forward_results_dict['h_next'][backward_iter]
                d_loss_d_h_next = forward_results_dict['d_loss_d_h_next'][backward_iter]

                # execute backward pass
                backward_results_dict_d_h_prev['d_loss_d_h_prev'][backward_iter], results_dict_iter = (
                    gru_cell_1.backward_combined(learning_rate, x, r, z, h_prev, h_candidate, h_next, d_loss_d_h_next) )

                # Accumulate gradients
                for key in backward_results_dict_total:
                    backward_results_dict_total[key] += results_dict_iter[key]
                        
            # we've reached the end of a time step sequence, so update weights:
            gru_cell_1.update_weights(backward_results_dict_total, learning_rate)
            backward_results_dict_total = {key: np.zeros_like(value) for key, value in backward_results_dict_total.items()} # Reset for the next time step sequence

        # Store or print the loss after each epoch
        losses.append(total_epoch_loss)
        print(f'Epoch {epoch+1}, Total Loss: {total_epoch_loss}')

    return losses

def main():
    # Main execution setup
    input_size = 1  # As series is 1-dimensional
    hidden_size = 3
    gru_cell_1 = gru_cell(input_size, hidden_size)

    # grader: uncomment here to run on a simple series
    # sometimes it fails/returns nan but it commonly succeeds and gets zero error with this simple function
    # conditions for success: hidden_size and learning rate work at 5 and 0.01, but you can try other values and they may well work too
    # use sequence length of 128 or more
    # Generate and train on a simple series
    #series_length = 100
    #series = generate_series(series_length)

    import os
    import pandas as pd

    file_path = 'C:\\stocks\\aapl.us.txt'
    #file_path = 'C:\\stocks\\tsla.us.txt'

    # Read the data into a DataFrame using pandas
    stock_data = pd.read_csv(
        file_path,
        delimiter=",",  # Comma-separated values
        usecols=["Close"]  # Extract only the 'Close' column
    )

    # Take the first 100 data points and convert to a list or NumPy array
    series = stock_data["Close"].head(100).values

    series = series.reshape(-1, 1)  # Reshape series to fit input size expected by GRU

    # Train the GRU
    epochs = 5
    learning_rate = 0.01
    sequence_length = 64
    losses = train_gru(gru_cell_1, series, epochs, learning_rate, sequence_length)

if __name__ == '__main__':
    main()