import pyautogui
import psutil
import time
import cv2
import torch
import numpy as np
import tkinter as tk

class GameAgent:
    def __init__(self, game_name):
        self.game_name = game_name
        self.game_process = None
        self.current_state = None
        self.last_score = None
        self.last_action = None
        self.last_state = None
        self.last_reward = None
        self.reward_sum = 0
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.memory = []
        self.batch_size = 32
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.create_model()
        self.actions = ['up', 'down', 'left', 'right', 'click']

    def create_model(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(6400, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 5)
        )
        model.load_state_dict(torch.load('model.pt'))
        model.eval()
        return model

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_game_window(self):
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            if self.game_name.lower() in proc.info['name'].lower():
                self.game_process = proc
                return proc.pid

    def get_state(self):
        hwnd = self.get_game_window()
        if hwnd is None:
            return None

        bbox = (0, 0, 800, 600)
        img = pyautogui.screenshot(region=bbox)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (80, 60))
        return img.flatten()

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state_tensor)
            action_idx = torch.argmax(q_values).item()
            action = self.actions[action_idx]

        self.last_action = action
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.reward_sum += reward
        if done:
            self.last_reward = self.reward_sum
            self.reward_sum = 0

        if len(self.memory) > 10000:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = []
        targets = []
        for i in batch:
            state, action, reward, next_state, done = self.memory[i]
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            q_values = self.model(state_tensor)
            next_q_values = self.model(next_state_tensor)
            if done:
                target = reward
            else:
                target = reward + self.discount_factor * torch.max(next_q_values)
                        q_values[action] = target
        states.append(state)
        targets.append(q_values)

    states_tensor = torch.tensor(states, dtype=torch.float32)
    targets_tensor = torch.stack(targets)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    optimizer.zero_grad()
    loss = torch.nn.MSELoss()(self.model(states_tensor), targets_tensor)
    loss.backward()
    optimizer.step()

def play(self):
    root = tk.Tk()
    root.title("Game Agent")
    canvas = tk.Canvas(root, width=800, height=600)
    canvas.pack()
    self.gui_loop(root, canvas)
    root.mainloop()

def gui_loop(self, root, canvas):
    while True:
        state = self.get_state()
        if state is not None:
            action = self.get_action(state)
            self.execute_action(action)

        if self.last_score is not None:
            canvas.create_text(20, 20, anchor=tk.NW, text="Score: {}".format(self.last_score), fill='white')

        canvas.update()

        time.sleep(0.1)

def execute_action(self, action):
    if action == 'up':
        pyautogui.press('up')
    elif action == 'down':
        pyautogui.press('down')
    elif action == 'left':
        pyautogui.press('left')
    elif action == 'right':
        pyautogui.press('right')
    elif action == 'click':
        pyautogui.click()

def save_model(self):
    torch.save(self.model.state_dict(), 'model.pt')
   
if name == 'main':
game_agent = GameAgent('your_game_name')
game_agent.play()
           
