import msgpack
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
import torch.nn as nn
from car_env import Environment


class DQN(nn.Module):
    def __init__(self, inp, hid, out, lr):
        
        super().__init__()

        self.inp = inp
        self.hid = hid
        self.out = out

        self.layer_stack = nn.Sequential(
            nn.Linear(inp, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, out)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x):
        return self.layer_stack(x)


class DQL():

    def __init__(self):
        self.device = 'cuda'
        self.gamma = .99
        self.epsilon = 1.0
        self.eps_min = .01
        self.eps_dec = .997
        self.lr = .001
        self.mem_size = 3000
        self.batch_size = 32
        self.mem_cntr = 0

        self.env = Environment()
        self.num_states = self.env.n_states
        self.num_actions = self.env.n_actions
        self.action_space = self.env.action_space

        self.Q_eval = DQN(self.num_states, 256, self.num_actions, self.lr).to(self.device)
        self.state_memory = np.zeros((self.mem_size, self.num_states), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, self.num_states), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1
    
    def choose_action(self, observation, test=False):
        if np.random.random() > self.epsilon or test:
            with torch.inference_mode():
                state = torch.tensor([observation]).type(torch.float).to(self.device)
                actions = self.Q_eval(state)
                action = torch.argmax(actions).item()
        else:
            action = random.randint(0, self.num_actions-1)
        
        return self.action_space[action]

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss_fn(q_target, q_eval).to(self.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        # self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def train(self, episodes):
        scores, eps_history = [], []
        avg_scores = []
        for _ in range(episodes):
            score = 0
            done = False
            observation = self.env.reset()
            steps = 0

            max_steps = 3000

            while not done and steps < max_steps:
                action = self.choose_action(observation)
                observation_, reward, done = self.env.step(action)
                score += reward

                self.store_transition(observation, action, reward, observation_, done)

                self.learn()

                observation = observation_
                steps += 1
            
            scores.append(score)
            eps_history.append(self.epsilon)

            avg_scores.append(np.mean(scores[-100:]))

            print(f'Episode {_} | Score: {score:.2f} | Average Score: {np.mean(scores[-100:]):.2f} | Epsilon: {self.epsilon:.2f}')
            if (_+1) % 1 == 0:
                x = [i+1 for i in range(episodes)]
                plt.figure(figsize=(10, 10))
                plt.scatter([i+1 for i in range(len(scores))], scores, color='orange')
                plt.plot(avg_scores)
                plt.plot(eps_history, color='yellow')
                plt.savefig('plot.png')
                plt.close()
            
            torch.save(self.Q_eval.state_dict(), 'dqn.pt')
            # if (_+1) % 200 == 0:
            #     self.test(30)

    def test(self, episodes, MODEL_PATH=None):

        if MODEL_PATH is not None:
            self.Q_eval.load_state_dict(torch.load(MODEL_PATH))

        orig_balance = 1000
        balance = orig_balance
        trades = 0
        inv = 100
        action_counter = [0 for i in range(self.num_actions)]

        for _ in range(episodes):
            score = 0
            done = False
            observation = self.env.reset()
            steps = 0

            while not done:
                action = self.choose_action(observation, test=True)
                action_counter[self.action_space.index(action)] += 1
                observation_, reward, done = self.env.step(action, test=True)

                if done:
                    break

                balance += inv * reward
                trades += 1

                score += reward

                observation = observation_
                steps += 1
            
            ppt = (balance - orig_balance) / (trades * inv) * 100

        print(f'Test | Balance: {balance} | PPT: {ppt:.4f}% | Action Counter: {action_counter}')




if __name__ == '__main__':
    d = DQL()
    d.train(10000)
    






