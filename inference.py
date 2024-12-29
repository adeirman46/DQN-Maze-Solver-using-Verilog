import numpy as np
import torch
import matplotlib.pyplot as plt
from environment import MazeEnvironment
from agent import Agent
import collections
import torch.nn as nn
import time

class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size, device = 'cuda'):
        indices = np.random.choice(len(self.memory), batch_size, replace = False)
        states, actions, next_states, rewards, isgameon = zip(*[self.memory[idx] 
                                                            for idx in indices])
        return torch.Tensor(states).type(torch.float).to(device), \
               torch.Tensor(actions).type(torch.long).to(device), \
               torch.Tensor(next_states).to(device), \
               torch.Tensor(rewards).to(device), torch.tensor(isgameon).to(device)

class fc_nn(nn.Module):
    def __init__(self, Ni, Nh1, Nh2, No=4):
        super().__init__()
        self.fc1 = nn.Linear(Ni, Nh1)
        self.fc2 = nn.Linear(Nh1, Nh2)
        self.fc3 = nn.Linear(Nh2, No)
        self.act = nn.ReLU()
        
    def forward(self, x, classification=False, additional_out=False):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        out = self.fc3(x)
        return out

def run_inference(model_path='best2.torch', device='cuda'):
    # Load maze and setup
    maze = np.load('maze_generator/maze.npy')
    initial_position = [0, 0]
    goal = [len(maze)-1, len(maze)-1]
    maze_env = MazeEnvironment(maze, initial_position, goal)
    
    # Setup buffer and agent
    buffer_capacity = 10000
    memory_buffer = ExperienceReplay(buffer_capacity)
    agent = Agent(maze=maze_env, memory_buffer=memory_buffer)
    
    # Setup network
    net = fc_nn(maze.size, maze.size, maze.size, 4)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    net.eval()

    # Setup matplotlib figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()  # Turn on interactive mode
    
    # Set up agent for inference
    agent.isgameon = True
    agent.use_softmax = False
    _ = agent.env.reset(0)

    # Main animation loop
    while agent.isgameon:
        ax.clear()
        ax.imshow(agent.env.maze, interpolation='none', aspect='equal', cmap='Greys')
        ax.plot(agent.env.goal[1], agent.env.goal[0], 'bs', markersize=4)
        ax.plot(agent.env.current_position[1], agent.env.current_position[0], 'rs', markersize=4)
        plt.xticks([])
        plt.yticks([])
        
        # Draw and pause to show the current state
        plt.draw()
        plt.pause(0.5)  # Pause for 0.5 seconds
        
        # Make the next move
        agent.make_a_move(net, 0)
    
    plt.ioff()  # Turn off interactive mode
    plt.show()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    run_inference(device=device)