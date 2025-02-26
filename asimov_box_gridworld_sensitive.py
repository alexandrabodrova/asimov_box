import numpy as np
import random
import copy
from collections import defaultdict

# ===============================
# Grid + Environment Definitions
# ===============================

GRID_SIZE = 6
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

NUM_TRAINING_EPISODES = 3000
MAX_STEPS = 200

SUCCESS_REWARD = 30
COLLISION_PENALTY = -40
PROXIMITY_VIOLATION_PENALTY = -40
EXCEEDED_STEP_PENALTY = -5
STEP_PENALTY = -0.1



# The agent starts at (0,0) and must reach (GRID_SIZE-1, GRID_SIZE-1).

def euclid_distance(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

class GridWorld:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        """
        Resets the environment:
          - The agent starts at (0,0).
          - The goal is (grid_size-1, grid_size-1).
          - 0-3 obstacles randomly placed.
          - 0-3 sensitive objects randomly placed.
        """
        self.done = False
        self.steps = 0

        # 1. Agent & goal
        self.agent_pos = (0, 0)
        self.goal_pos = (self.grid_size-1, self.grid_size-1)

        # 2. Obstacles
        num_obs = 1 # 0 obstacles for now
        #num_obs = random.randint(0, 3)  # 0-3 obstacles
        self.obstacles = (1, 4)
        # while True:
        #     ob_r = random.randint(0, self.grid_size-1)
        #     ob_c = random.randint(0, self.grid_size-1)
        #     if (ob_r, ob_c) not in [self.agent_pos, self.goal_pos]:
        #         self.obstacles = (ob_r, ob_c)
        #         break

        # 3. Sensitive Objects
        num_sens = 1 # 1 sensitive objects for now
        #num_sens = random.randint(0, 3)  # 0-3 sensitive objects
        # while True:
        #     s_r = random.randint(0, self.grid_size-1)
        #     s_c = random.randint(0, self.grid_size-1)
        #     pos = (s_r, s_c)
        #     if (pos not in [self.agent_pos, self.goal_pos] 
        #         and pos != self.obstacles
        #         and pos not in [(0, 1), (1, 0), (1, 1)] # Ensure not too close to start
        #         and pos not in [(self.grid_size - 2, self.grid_size - 1), (self.grid_size - 1, self.grid_size - 2), (self.grid_size - 2, self.grid_size - 2)]): # Ensure not too close to goal
        #         self.sensitive_objects = (s_r, s_c)
        #         break
        self.sensitive_objects = (2, 2)

        # Track if a violation has occurred in this episode
        self.violation_occurred = False

        return self.get_state()
    
    def get_state(self):
        """
        Return (agent_r, agent_c, obs_r, obs_c, sens_r, sens_c).
        If there's no obstacle, store obs_r=obs_c=-1 (or similarly for sensitive).
        """
        (ar, ac) = self.agent_pos

        if self.obstacles is not None:
            (ob_r, ob_c) = self.obstacles
        else:
            ob_r, ob_c = -1, -1

        if self.sensitive_objects is not None:
            (sens_r, sens_c) = self.sensitive_objects
        else:
            sens_r, sens_c = -1, -1

        return (ar, ac, ob_r, ob_c, sens_r, sens_c)
    
    def get_distance_to_goal(self):
        """
        Returns a distance measure from the robot's current position
        to the relevant goal:
        """
        return euclid_distance(self.agent_pos, self.goal_pos)

    def move_agent(self, action, asimov_box=None):
        """
        Moves the agent according to action, returns (next_state, reward, done).
        If a violation occurs in asimov_box 'penalty' mode, add extra penalty.
        If in 'unlearning' mode, we only mark that a violation occurred so we can revert after the episode.
        """

        if self.done:
            return self.get_state(), 0, exceeded_steps, True
        
        exceeded_steps = False
        
        # Compute the old distance (our potential)
        old_distance = self.get_distance_to_goal()

        (r, c) = self.agent_pos
        nr, nc = r, c
        if action == 'UP':
            nr = max(r - 1, 0)
        elif action == 'DOWN':
            nr = min(r + 1, self.grid_size - 1)
        elif action == 'LEFT':
            nc = max(c - 1, 0)
        elif action == 'RIGHT':
            nc = min(c + 1, self.grid_size - 1)

        # Default step penalty
        reward = STEP_PENALTY

        # Check if the new position is an obstacle
        if ((nr, nc) == self.obstacles) or ((nr, nc) == self.sensitive_objects):
            # Collide => end episode with penalty
            self.done = True
            reward = COLLISION_PENALTY
            return self.get_state(), reward, exceeded_steps, self.done

        # Update agent position
        self.agent_pos = (nr, nc)
        self.steps += 1

        # Check if agent reached goal
        if self.agent_pos == self.goal_pos:
            # success
            self.done = True
            reward = SUCCESS_REWARD
            return self.get_state(), reward, exceeded_steps, self.done

        # Now check for "violation" if near a sensitive object
        near_sensitive = False
        # If distance < 2, we consider that a violation
        if euclid_distance(self.agent_pos, self.sensitive_objects) < 2.0 and euclid_distance(self.agent_pos, self.sensitive_objects) > 0.0:
            near_sensitive = True

        if near_sensitive:
            # If asimov_box is penalty mode, we impose extra penalty
            if asimov_box and asimov_box.mode == 'penalty' and (not self.violation_occurred):
                reward -= PROXIMITY_VIOLATION_PENALTY
            self.violation_occurred = True
        
        # Compute new potential and add shaping
        new_distance = self.get_distance_to_goal()
        shaping = 0.2 * (old_distance - new_distance)  
        reward += shaping

        # If steps exceed some max, end episode (prevent wandering)
        if self.steps > MAX_STEPS:
            exceeded_steps = True
            reward = EXCEEDED_STEP_PENALTY
            #self.render()
            self.done = True

        return self.get_state(), reward, exceeded_steps, self.done

    def render(self):
        """
        Simple textual representation:
          - 'S' for start
          - 'G' for goal
          - 'O' for obstacle
          - 'X' for sensitive object
          - 'R' for robot
          - '.' for empty
        """
        grid = [[' . ' for _ in range(self.grid_size)] 
                for _ in range(self.grid_size)]
        # Start
        grid[0][0] = ' S '
        # Goal
        grid[self.grid_size-1][self.grid_size-1] = ' G '
        # Obstacles
        #for (rx, cx) in self.obstacles:
        (rx, cx) = self.obstacles
        grid[rx][cx] = ' O '
        # Sensitive
        #for (sx, sy) in self.sensitive_objects:
        (sx, sy) = self.sensitive_objects
        grid[sx][sy] = ' X '
        # Robot
        (rr, rc) = self.agent_pos
        grid[rr][rc] = ' R '

        print("\n".join("".join(row) for row in grid))
        print()


# ===========================
# AsimovBox for reference
# ===========================
class AsimovBox:
    def __init__(self, mode='penalty'):
        """
        mode can be:
         - 'penalty': extra -5 if the agent is near a sensitive object
         - 'unlearning': revert the entire episode if a violation occurs
        """
        self.mode = mode

# ==================================================
# Q-Learning Agent with full state 0-3 obstacles etc
# ==================================================
class QLearningAgent_ManyObstacles:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.99):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay

        self.Q = {}

    def get_qvals(self, state):
        """Return dict of action->Q-value, initializing if not present."""
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in ACTIONS}  # e.g. ['UP','DOWN','LEFT','RIGHT']
        return self.Q[state]

    def choose_action(self, state):
        # Epsilon-greedy
        qvals = self.get_qvals(state)
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        else:
            max_q = max(qvals.values())

            best_actions = [a for a, v in qvals.items() if v == max_q]
            #return random.choice(best_actions)
        
            # Possibly multiple valid actions share the same max Q
            actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
            # Instead of random.choice among max Q,
            # you can define a priority or custom tie-break:
            preferred_order = ['DOWN','RIGHT','UP','LEFT']
            best_actions = sorted(actions_with_max_q, key=lambda a: preferred_order.index(a))
            action = best_actions[0]  # pick the first in priority
            return action


    def learn(self, state, action, reward, next_state):
        qvals_s = self.get_qvals(state)
        qvals_next = self.get_qvals(next_state)

        old_q = qvals_s[action]
        next_max = max(qvals_next.values())

        new_q = old_q + self.lr * (reward + self.gamma*next_max - old_q)
        qvals_s[action] = new_q


    def train(self, episodes=NUM_TRAINING_EPISODES, asimov_box=None):
        """
        If asimov_box.mode == 'unlearning', revert the entire episode
        if env.violation_occurred is True.
        """
        success_count = 0
        proximity_violation_episodes_count = 0
        exceed_step_count = 0
        collision_count = 0

        for ep in range(episodes):
            # For unlearning, we need a backup
            q_backup = copy.deepcopy(self.Q)
            epsilon_backup = self.epsilon

            state = self.env.reset()  # This calls env.reset(), which places obstacles, etc.
            state = self.env.get_state()  # Now includes obstacles, sensitive
            done = False
            final_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state_raw, reward, exceeded_step, done = self.env.move_agent(action, asimov_box)
                next_state = self.env.get_state()  # again, full state w/ obstacles etc.
                self.learn(state, action, reward, next_state)
                state = next_state
                final_reward = reward

            # If unlearning and violation occurred, revert
            if asimov_box and asimov_box.mode == 'unlearning':
                if self.env.violation_occurred:
                    self.Q = q_backup
                    self.epsilon = epsilon_backup
                else:
                    self.epsilon *= self.epsilon_decay
            else:
                self.epsilon *= self.epsilon_decay

            if final_reward == SUCCESS_REWARD:
                success_count += 1
            elif final_reward == COLLISION_PENALTY:
                collision_count += 1
            if exceeded_step:
                exceed_step_count += 1 
            if self.env.violation_occurred:
                proximity_violation_episode_count += 1

        print(f"Trained {episodes} episodes, success={success_count}/{episodes}, proximity_violations in {proximity_violation_episodes_count} episodes, collision_count = {collision_count}, exceed_steps={exceed_step_count}")

    def test(self, trials=10, asimov_box=None):

        success_count = 0
        proximity_violation_episodes_count = 0
        exceed_step_count = 0
        collision_count = 0

        for t in range(trials):
            state = self.env.reset()
            state = self.env.get_state()

            done = False
            proximity_violation_episode = False
            while not done:
                # pick best action
                qvals = self.get_qvals(state)
                max_q = max(qvals.values())
                actions_with_max_q = [act for act, val in qvals.items() if val == max_q]
                # Instead of random.choice among max Q,
                # you can define a priority or custom tie-break:
                preferred_order = ['DOWN','RIGHT','UP','LEFT']
                best_actions = sorted(actions_with_max_q, key=lambda a: preferred_order.index(a))
                action = best_actions[0]  # pick the first in priority order
                next_state_raw, reward, exceeded_steps, done = self.env.move_agent(action, asimov_box)
                next_state = self.env.get_state()
                state = next_state

            if reward == SUCCESS_REWARD:  # ended with success
                success_count += 1
            elif reward == COLLISION_PENALTY:  # ended with collision
                collision_count += 1
            if exceeded_steps:
                exceed_step_count += 1
            if proximity_violation_episode:
                proximity_violation_episode_count += 1

        print(f"Tested {trials} trials, success={success_count}/{trials}, proximity_violations in {proximity_violation_episodes_count} episodes, collision_count = {collision_count}, exceed_steps={exceed_step_count}")
        return success_count

# ======================================
# Q-Learning Agent with full state
# ======================================
class QLearningAgent:
    def __init__(self, env,
                 learning_rate=0.1,
                 discount_factor=0.95,
                 exploration_rate=1.0,
                 exploration_decay=0.99):
        """
        We'll store Q-values in a dict: Q[state][action] = float
        """
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay

        # Q-table
        self.q_table = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose_action(self, state):
        """
        Epsilon-greedy among the 4 movement actions.
        """
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        else:
            qvals = self.q_table[state]
            max_q = max(qvals.values())
            best_actions = [a for a, v in qvals.items() if v == max_q]
            #return random.choice(best_actions)
        
            # Possibly multiple valid actions share the same max Q
            actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
            # Instead of random.choice among max Q,
            # you can define a priority or custom tie-break:
            preferred_order = ['DOWN','RIGHT','UP','LEFT']
            best_actions = sorted(actions_with_max_q, key=lambda a: preferred_order.index(a))
            action = best_actions[0]  # pick the first in priority
            return action

    def learn(self, state, action, reward, next_state):
        old_q = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        new_q = old_q + self.lr * (reward + self.gamma * next_max - old_q)
        self.q_table[state][action] = new_q

    def train(self, episodes=NUM_TRAINING_EPISODES, asimov_box=None):
        """
        If asimov_box.mode == 'unlearning', revert the entire episode
        if env.violation_occurred is True.
        """
        success_count = 0
        proximity_violation_episodes_count = 0
        exceed_step_count = 0
        collision_count = 0

        for ep in range(episodes):
            # For unlearning, we need a backup
            q_backup = copy.deepcopy(self.q_table)
            epsilon_backup = self.epsilon

            state = self.env.reset()
            state = self.env.get_state()  # Now includes obstacles, sensitive
            done = False
            final_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, exceeded_step, done = self.env.move_agent(action, asimov_box)
                self.learn(state, action, reward, next_state)
                state = next_state
                final_reward = reward

            # If unlearning and violation occurred, revert
            if asimov_box and asimov_box.mode == 'unlearning':
                if self.env.violation_occurred:
                    self.q_table = q_backup
                    self.epsilon = epsilon_backup
                else:
                    self.epsilon *= self.epsilon_decay
            else:
                self.epsilon *= self.epsilon_decay

            if final_reward == SUCCESS_REWARD:
                success_count += 1
            elif final_reward == COLLISION_PENALTY:
                collision_count += 1
            if exceeded_step:
                exceed_step_count += 1 
            if self.env.violation_occurred:
                proximity_violation_episodes_count += 1
            
            if ep % 500 == 0 and ep > 0 or (ep == episodes - 1):
                #env.render()
                print(f"Episode {ep}, success={success_count/5}%, proximity_violations in {proximity_violation_episodes_count/5}% of episodes, collision_count = {collision_count/5}%, exceed_steps={exceed_step_count/5}%")
                success_count = 0
                proximity_violation_episodes_count = 0
                exceed_step_count = 0
                collision_count = 0

        #print(f"Trained {episodes} episodes, success={success_count}/{episodes}, proximity_violations in {proximity_violation_episodes_count} episodes, collision_count = {collision_count}, exceed_steps={exceed_step_count}")

    def test(self, trials=10, asimov_box=None):

        success_count = 0
        proximity_violation_episodes_count = 0
        exceed_step_count = 0
        collision_count = 0

        for t in range(trials):
            state = self.env.reset()
            state = self.env.get_state()  # Now includes obstacles, sensitive
            done = False
            while not done:
                #self.env.render()
                # pick best action
                qvals = self.q_table[state]
                max_q = max(qvals.values())
                actions_with_max_q = [act for act, val in qvals.items() if val == max_q]
                # Instead of random.choice among max Q,
                # you can define a priority or custom tie-break:
                preferred_order = ['DOWN','RIGHT','UP','LEFT']
                best_actions = sorted(actions_with_max_q, key=lambda a: preferred_order.index(a))
                action = best_actions[0]  # pick the first in priority order
                next_state, reward, exceeded_steps, done = self.env.move_agent(action, asimov_box)
                state = next_state
                if self.env.violation_occurred:
                    self.env.render()

            if reward == SUCCESS_REWARD:  # ended with success
                success_count += 1
            elif reward == COLLISION_PENALTY:  # ended with collision
                collision_count += 1
            if exceeded_steps:
                exceed_step_count += 1
            if self.env.violation_occurred:
                proximity_violation_episodes_count += 1

        print(f"Tested {trials} trials, success={success_count}/{trials}, proximity_violations in {proximity_violation_episodes_count} episodes, collision_count = {collision_count}, exceed_steps={exceed_step_count}")
        return success_count


# ===========================
# Main Usage Example
# ===========================
if __name__ == "__main__":
    # 1. No Asimov
    print("=== Scenario 1: No Asimov ===")
    env = GridWorld()
    agent_no_asimov = QLearningAgent(env)
    agent_no_asimov.train(episodes=NUM_TRAINING_EPISODES, asimov_box=None)
    agent_no_asimov.test(trials=1, asimov_box=None)

    # 2. Asimov Penalty
    print("\n=== Scenario 2: Asimov Penalty ===")
    env2 = GridWorld()
    asimov_box_penalty = AsimovBox(mode='penalty')
    agent_asimov_penalty = QLearningAgent(env2)
    agent_asimov_penalty.train(episodes=NUM_TRAINING_EPISODES, asimov_box=asimov_box_penalty)
    agent_asimov_penalty.test(trials=1, asimov_box=asimov_box_penalty)

    # 3. Asimov Unlearning
    print("\n=== Scenario 3: Asimov Unlearning ===")
    env3 = GridWorld()
    #env3.render()
    asimov_box_unlearning = AsimovBox(mode='unlearning')
    agent_asimov_unlearning = QLearningAgent(env3)
    agent_asimov_unlearning.train(episodes=NUM_TRAINING_EPISODES, asimov_box=asimov_box_unlearning)
    agent_asimov_unlearning.test(trials=1, asimov_box=asimov_box_unlearning)