import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, h1_size = 256, h2_size = 256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Total number of actions
            seed (int): Random seed
        """
        super(ANetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action probabilities."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.Softmax(self.fc3(x))
        return x