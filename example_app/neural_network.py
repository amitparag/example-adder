"""
Claas to instantiate a neural network that will learn Loss function described in loss_function.py.
"""
import torch as T

class LossApproximator(T.nn.Module):

    def __init__(self, input_dimensions:int = 3, output_dimensions:int = 1, 
                hidden_units:list = [32,32,5], activation:T.nn.modules.activation = T.nn.Sigmoid()):
        super(LossApproximator, self).__init__()
        
        
        """
        A simple three hidden layered neural network that has two additional functions: gradient and hessian
        """

        self.input_dimensions   = input_dimensions
        self.output_dimensions  = output_dimensions
        self.fc1_dims           = hidden_units[0]
        self.fc2_dims           = hidden_units[1]
        self.fc3_dims           = hidden_units[2]
        self.activation         = activation


        #........... Structure. 4 fully connected layers
        # fc1 --> fc2 --> fc3 --> fc4
        self.fc1 = T.nn.Linear(self.input_dimensions, self.fc1_dims)
        self.fc2 = T.nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = T.nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = T.nn.Linear(self.fc3_dims, self.output_dimensions)

        # Weight Initialization protocol
        T.nn.init.kaiming_uniform_(self.fc1.weight)
        T.nn.init.kaiming_uniform_(self.fc2.weight)
        T.nn.init.kaiming_uniform_(self.fc3.weight)
        T.nn.init.kaiming_uniform_(self.fc4.weight)

        # Bias Initialization protocol
        self.fc1.bias.data.fill_(0.001)
        self.fc2.bias.data.fill_(0.003)
        self.fc3.bias.data.fill_(0.003)
        self.fc4.bias.data.fill_(0.003)


    def forward(self, state):
        state    = self.activation(self.fc1(state))
        state    = self.activation(self.fc2(state))
        state    = self.activation(self.fc3(state)) 
        value    = self.fc4(state)
        return value
    
    @T.jit.ignore # This is needed for scipting to torch_script. Otherwise gradient function cannot be used in c++ version
    def gradient(self, state):
        """
        Calculates the gradient of the neural network prediction with respect to a single input
        """   
        j = T.autograd.functional.jacobian(self.forward, state).squeeze()
        return j

    @T.jit.ignore 
    def hessian(self, x):
        """
        Calculate and return the hessian of the neural network prediction with respect to a single input
        """
        
        h = T.autograd.functional.hessian(self.forward, x).squeeze()
        return h
    
    
if __name__=="__main__":
    import torch
    loss_net = LossApproximator()
    print(loss_net)
    
    scripted_loss_network = torch.jit.script(loss_net)
    try:
        assert 'gradient' in scripted_loss_network.__dict__     # Check if gradient has been scripted
        assert 'hessian'  in scripted_loss_network.__dict__     # Check if hessian has been scripted
    except AssertionError as error:
        print("Gradient and Hessian not scripted")
        
    scripted_loss_network.save("network.pt")


