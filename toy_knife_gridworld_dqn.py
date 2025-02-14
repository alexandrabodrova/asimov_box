import numpy as np
import random
import copy
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim

# ======================
# ENVIRONMENT CODE
# ======================

GRID_SIZE = 6
LIVING_ROOM = [(0,0), (0,1), (1,0), (1,1)]  # top-left 2x2
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK', 'DROP']
OBJECT_TYPES = ['knife', 'toy']
SUCCESSFUL_DROP_REWARD = 5
COLLISION_PENALTY = -5
SUCCESSFUL_PICKUP_REWARD = 3
UNSUCCESSFUL_PICKUP_PENALTY = -1
WRONG_DROP_PENALTY = -3
PROXIMITY_PENALTY = -2

def euclid_distance(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

class GridWorld:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.reset()

    def reset(self):
        # ... identical to your existing reset() ...
        import numpy as np

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Define living room and place robot
        self.living_room = LIVING_ROOM
        self.robot_pos = (0, 0)

        # Place kitchen in corner
        corners = [(self.grid_size-2, self.grid_size-2)]
        # self.kitchen_corner = random.choice(corners)
        # self.kitchen = [(self.kitchen_corner[0] + x, self.kitchen_corner[1] + y) 
        #                 for x in range(2) for y in range(2)]
        self.kitchen_corner = (2, 2)
        self.kitchen = [(2,2), (2,3), (3,2), (3,3)]
        self.object_pos = (2,2)  # always top-left corner of kitchen
        self.humans = [(2,1)]  # just one known position
        self.obstacles = []

        # Place object
        self.object_type = random.choice(OBJECT_TYPES)
        self.object_pos = random.choice(self.kitchen)
        self.grid[self.object_pos] = 1

        # Place 1 human
        self.humans = []
        while True:
            pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if (pos not in self.living_room and pos not in self.kitchen 
                and pos != self.object_pos):
                self.humans.append(pos)
                self.grid[pos] = 2
                break

        # Place 1-2 obstacles
        # self.obstacles = []
        # num_obstacles = random.randint(1, 2)
        # for _ in range(num_obstacles):
        #     while True:
        #         pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        #         if (pos not in self.living_room and pos not in self.kitchen 
        #             and pos != self.object_pos and pos not in self.humans 
        #             and pos not in self.obstacles):
        #             self.obstacles.append(pos)
        #             self.grid[pos] = 3
        #             break

        self.robot_carrying = False
        self.robot_carrying_type = None
        self.steps = 0
        self.done = False
        self.proximity_violation_in_this_episode = False

        return self.get_state()

    def get_state(self):
        """
        (robot_pos, object_pos, carrying_bool, carry_type_idx)
        """
        if self.robot_carrying:
            object_pos = (-1, -1)
            carry_type_idx = OBJECT_TYPES.index(self.robot_carrying_type)
        else:
            object_pos = self.object_pos
            carry_type_idx = -1
        return (self.robot_pos, object_pos, self.robot_carrying, carry_type_idx)

    def get_distance_to_goal(self):
        """
        Returns a distance measure from the robot's current position
        to the relevant goal:
        - If not carrying, the goal is the object's position
        - If carrying, the goal is the nearest living room cell
        """
        if not self.robot_carrying:
            # Distance from robot to the object
            return euclid_distance(self.robot_pos, self.object_pos)
        else:
            # Distance from robot to the NEAREST living room cell
            min_dist = float('inf')
            for lr_cell in self.living_room:
                d = euclid_distance(self.robot_pos, lr_cell)
                if d < min_dist:
                    min_dist = d
            return min_dist

    def move_robot(self, action, asimov_box=None):
        step_overflow = False
        if self.done:
            return self.get_state(), 0, self.done, step_overflow

         # 1. Compute the old distance (our potential)
        old_distance = self.get_distance_to_goal()

        # Normal movement/reward logic ...
        #asimov_box_penalty = (asimov_box is not None and asimov_box.mode == 'penalty')
        x, y = self.robot_pos
        new_x, new_y = x, y
        reward = -0.01  # small step penalty by default

        x, y = self.robot_pos
        new_x, new_y = x, y

        # Basic movement
        if action == 'UP':
            new_x = max(x - 1, 0)
        elif action == 'DOWN':
            new_x = min(x + 1, self.grid_size - 1)
        elif action == 'LEFT':
            new_y = max(y - 1, 0)
        elif action == 'RIGHT':
            new_y = min(y + 1, self.grid_size - 1)
        elif action == 'PICK':
            # Attempt pick
            if (not self.robot_carrying) and (self.robot_pos == self.object_pos):
                self.robot_carrying = True
                self.robot_carrying_type = self.object_type
                self.grid[self.object_pos] = 0
                reward = SUCCESSFUL_PICKUP_REWARD
                return self.get_state(), reward, self.done, step_overflow
            else:
                reward = UNSUCCESSFUL_PICKUP_PENALTY
                return self.get_state(), reward, self.done, step_overflow
        elif action == 'DROP':
            # Attempt drop
            if self.robot_carrying and (self.robot_pos in self.living_room):
                self.robot_carrying = False
                self.robot_carrying_type = None
                self.grid[self.robot_pos] = 1
                self.done = True
                reward = SUCCESSFUL_DROP_REWARD
                return self.get_state(), reward, self.done, step_overflow
            else:
                self.done = True
                reward = WRONG_DROP_PENALTY
                return self.get_state(), reward, self.done, step_overflow

        new_pos = (new_x, new_y)
        if new_pos in self.obstacles or new_pos in self.humans:
            self.done = True
            reward = COLLISION_PENALTY
            return self.get_state(), reward, self.done, step_overflow

        # Actually move
        self.robot_pos = new_pos
        self.steps += 1

        # Check proximity if carrying a knife
        proximity_violation = False
        if self.robot_carrying and self.robot_carrying_type == 'knife':
            for hpos in self.humans:
                if euclid_distance(self.robot_pos, hpos) < 2:
                    proximity_violation = True
                    self.proximity_violation_in_this_episode = True

        # Penalty if Asimov penalty mode
        if asimov_box and asimov_box.mode == 'penalty' and proximity_violation:
            reward -= PROXIMITY_PENALTY

        # optional shaping if desired
        # 2. Compute new potential and add shaping
        new_distance = self.get_distance_to_goal()
        shaping = 1 * (old_distance - new_distance)  
        reward += shaping
        # shaping = ...
        # reward += shaping

        if self.steps > 200:
            step_overflow = True
            self.done = True

        return self.get_state(), reward, self.done, step_overflow


class AsimovBox:
    def __init__(self, mode='penalty'):
        self.mode = mode


# ======================
# DQN IMPLEMENTATION
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def state_to_tensor(state):
    """
    Convert state tuple -> PyTorch float tensor.
    state = (robot_pos, object_pos, carrying_bool, carry_type_idx)
    We'll flatten them into a numeric vector:
       [robot_x, robot_y, object_x, object_y, carrying_flag, object_type_idx]
    If carrying, object_pos = (-1, -1).
    If not carrying, carry_type_idx = -1
    We'll just feed these as floats. 
    """
    (rx, ry), (ox, oy), carrying_bool, carry_idx = state
    # Convert booleans to 0/1
    cflag = 1.0 if carrying_bool else 0.0
    v = [rx, ry, ox, oy, cflag, float(carry_idx)]
    return torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)

class QNetwork(nn.Module):
    """
    A small feedforward network that takes state_to_tensor
    dimension = 6, outputs Q-values for each of the 6 possible actions.
    """
    def __init__(self, state_dim=6, action_dim=6):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # no activation => raw Q-values

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(
        self,
        env,
        learning_rate=1e-4,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        replay_capacity=10000,
        batch_size=32,
        update_target_every=100,
    ):
        self.env = env

        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = 0.01
        self.batch_size = batch_size

        # Main Q-network
        self.qnet = QNetwork().to(device)
        # Target Q-network
        self.target_qnet = QNetwork().to(device)
        self.target_qnet.load_state_dict(self.qnet.state_dict())

        self.optim = optim.Adam(self.qnet.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        self.update_target_every = update_target_every
        self.train_step_count = 0

    def choose_action(self, state):
        """
        Epsilon-greedy action selection. 
        We must also filter out invalid actions if carrying or not carrying.
        """
        (robot_pos, object_pos, carrying_bool, carry_idx) = state

        if carrying_bool:
            valid_actions = ['UP','DOWN','LEFT','RIGHT','DROP']
        else:
            valid_actions = ['UP','DOWN','LEFT','RIGHT','PICK']

        # Epsilon random
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            # Evaluate Q-network
            state_tensor = state_to_tensor(state)  # shape [1,6]
            with torch.no_grad():
                qvals = self.qnet(state_tensor)  # shape [1,6]
            qvals = qvals.cpu().numpy()[0]  # shape [6]

            # We want the best among valid_actions
            valid_idx = [ACTIONS.index(a) for a in valid_actions]
            # find the best action index
            best_idx = max(valid_idx, key=lambda i: qvals[i])
            return ACTIONS[best_idx]

    def update_network(self):
        """
        Sample from replay and do a DQN one-step update.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # not enough data yet

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        state_tensors = []
        next_state_tensors = []
        for s, ns in zip(states, next_states):
            state_tensors.append(state_to_tensor(s))       # shape [1,6]
            next_state_tensors.append(state_to_tensor(ns)) # shape [1,6]
        state_tensors = torch.cat(state_tensors, dim=0)         # shape [batch,6]
        next_state_tensors = torch.cat(next_state_tensors, dim=0) # shape [batch,6]

        # Action indices
        action_indices = [ACTIONS.index(a) for a in actions]
        action_tensor = torch.tensor(action_indices, dtype=torch.long, device=device)

        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        done_tensor = torch.tensor(dones, dtype=torch.float32, device=device)

        # Q(s,a) from current network
        q_values = self.qnet(state_tensors)  # shape [batch,6]
        # gather the chosen actions
        q_a = q_values.gather(1, action_tensor.view(-1,1)).squeeze(1)  # shape [batch]

        # Q target
        with torch.no_grad():
            # Next state Q-values from target network
            q_next = self.target_qnet(next_state_tensors)  # shape [batch,6]
            # get max over all next actions
            q_next_max = q_next.max(dim=1)[0]
            # if done, 0 else gamma * q_next_max
            q_target = reward_tensor + (1 - done_tensor) * self.gamma * q_next_max

        # Compute loss (MSE)
        loss = nn.MSELoss()(q_a, q_target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.train_step_count += 1
        # Periodically update target network
        if self.train_step_count % self.update_target_every == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())

    def train(self, episodes=3000, asimov_box=None):
        """
        Similar structure to your original train:
         - For each episode, run until done
         - Choose action, step environment
         - Store transitions in replay buffer
         - Update network with mini-batches
         - Decay epsilon
        """
        overall_success_count = 0
        success_count = 0
        proximity_violation = False
        proximity_violation_count = 0

        for ep in range(episodes):
            state = self.env.reset()
            done = False
            step_count = 0
            final_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, overflow = self.env.move_robot(action, asimov_box)
                if reward == PROXIMITY_PENALTY:
                    proximity_violation = True
                step_count += 1
                final_reward = reward

                # Store in replay
                self.replay_buffer.push(state, action, reward, next_state, done)

                # One step of DQN update
                self.update_network()

                state = next_state

            # Epsilon decay after each episode
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Check success
            if final_reward == SUCCESSFUL_DROP_REWARD:
                success_count += 1
                overall_success_count += 1
            if proximity_violation:
                proximity_violation_count += 1

            if (ep+1) % 100 == 0:
                print(f"Episode {ep+1}/{episodes}, success in these 100 episodes = {success_count}, proximity violation = {proximity_violation_count}")
                success_count = 0
                proximity_violation_count = 0

        print(f"Training complete. Total success = {overall_success_count}/{episodes}")

    def test(self, trials=10, asimov_box=None):
        """
        Similar to your original test, but use the neural net for action selection with e=0.
        """
        old_epsilon = self.epsilon
        self.epsilon = 0.0  # purely greedy

        success = 0
        for t in range(trials):
            state = self.env.reset()
            done = False
            step_count = 0
            final_reward = 0
            proximity_violation = False
            proximity_violation_count = 0
            while not done and step_count < 200:
                step_count += 1
                action = self.choose_action(state)  # now chooses best action always
                next_state, reward, done, overflow = self.env.move_robot(action, asimov_box)
                if reward == PROXIMITY_PENALTY:
                    proximity_violation = True
                    proximity_violation_count += 1
                state = next_state
                final_reward = reward

            if final_reward == SUCCESSFUL_DROP_REWARD:
                success += 1
            print(f"Test trial {t} ended with reward {final_reward}, steps={step_count}, proximity violation={proximity_violation}")

        self.epsilon = old_epsilon
        print(f"Test success = {success}/{trials}, proximity violation = {proximity_violation_count}")
        return success


# ======================
# EXAMPLE USAGE
# ======================
if __name__ == "__main__":
    # Reuse your GridWorld, AsimovBox, etc.
    env = GridWorld()
    box_penalty = AsimovBox(mode='penalty')

    # Create a DQNAgent
    agent = DQNAgent(
        env,
        learning_rate=1e-3,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995,
        replay_capacity=10000,
        batch_size=64,
        update_target_every=200,
    )

    ### No Asimov Box

    # Train
    print("Without Asimov Box")
    agent.train(episodes=1000, asimov_box=None)

    # Test
    agent.test(trials=10, asimov_box=None)

    ### With Asimov Box Penalty
    # Train
    print("With Asimov Box Penalty")
    agent.train(episodes=1000, asimov_box=box_penalty)

    # Test
    agent.test(trials=10, asimov_box=box_penalty)
