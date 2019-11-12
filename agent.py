import numpy as np
import random
from collections import namedtuple, deque

from model import AgentNetwork
from memory import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Agent-Network
        ## TODO: Initialize your action network here
        "*** YOUR CODE HERE ***"
        self.network = AgentNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)
        self.network.train()

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0, get_prob = False):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state)
        self.network.train()
        
        if get_prob:
            return action_values.cpu().data.numpy()
            
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def discount_rewards(self, rewards, gamma=0.99):
        r = np.array([gamma**i * rewards[i] 
                      for i in range(len(rewards))])
        # Reverse the array direction for cumsum and then
        # revert back to the original order
        r = r[::-1].cumsum()[::-1]
        return r - r.mean()
     
    def learn(self, experiences, gamma = GAMMA):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        rewards = self.discount_rewards(rewards);

        ## TODO: compute and minimize the loss using REINFORCE
        "*** YOUR CODE HERE ***"
        self.optimizer.zero_grad()
        state_tensor = torch.FloatTensor(states)
        reward_tensor = torch.FloatTensor(rewards)
        action_tensor = torch.LongTensor(actions)
        
        # Calculate loss
        logprob = torch.log(self.network.forward(state_tensor))
        selected_logprobs = reward_tensor * logprob[np.arange(len(action_tensor)), action_tensor]
        loss = -selected_logprobs.mean()
        
        # Calculate gradients
        loss.backward()
        # Apply gradients
        self.optimizer.step()
        
        
                             
