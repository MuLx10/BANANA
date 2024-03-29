import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, h1_size = 16, h2_size = 16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Total number of actions
            seed (int): Random seed
        """
        super(AgentNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.fc1 = nn.Linear(state_size, h1_size)
        #self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action probabilities."""
        x = F.relu(self.fc1(state))
        #x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x