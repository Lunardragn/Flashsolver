import cv2
import numpy as np
import win32gui
import win32api
import tensorflow as tf
import random

# define the game window title
GAME_WINDOW_TITLE = "Flash Game"


# manually select the game window
def select_game_window():
    windows = []
    win32gui.EnumWindows(lambda hwnd, windows:
                         windows.append(hwnd) if win32gui.IsWindowVisible(hwnd) and len(
                             win32gui.GetWindowText(hwnd)) > 0 else None, windows)
    game_window = None
    while game_window is None:
        print("Select the window to play the game and press enter...")
        input()
        for hwnd in windows:
            if GAME_WINDOW_TITLE in win32gui.GetWindowText(hwnd):
                game_window = hwnd
                break
        if game_window is None:
            print("No window found with title:", GAME_WINDOW_TITLE)
    print("Window selected:", GAME_WINDOW_TITLE)
    return game_window


# get the screenshot of the game window
def get_screen(game_window):
    left, top, right, bottom = win32gui.GetWindowRect(game_window)
    width = right - left
    height = bottom - top
    screenshot = np.array(cv2.cvtColor(np.array(win32gui.GetWindowDC(game_window)), cv2.COLOR_BGR2RGB))
    return screenshot, width, height


# define the deep-Q neural network
class DeepQNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(DeepQNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu', input_shape=state_size)
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.fc3 = tf.keras.layers.Dense(action_size, activation='linear')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        q_values = self.fc3(x)
        return q_values


# define the replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, batch_size):
        return np.reshape(np.array(random.sample(self.buffer, batch_size)), [batch_size, 5])


# define the deep-Q learning algorithm
class DeepQLearning:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.batch_size = 64
        self.memory_size = 1000000
        self.replay_buffer = ReplayBuffer(self.memory_size)
        self.model = DeepQNetwork(self.state_size, self.action_size)
        self.target_model = DeepQNetwork(self.state_size, self.action_size)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(self.action_size)
            return action


def train(self):
    if len(self.replay_buffer.buffer) < self.batch_size:
        return
    mini_batch = self.replay_buffer.sample(self.batch_size)
    states = np.array([transition[0] for transition in mini_batch])
    actions = np.array([transition[1] for transition in mini_batch])
    rewards = np.array([transition[2] for transition in mini_batch])
    next_states = np.array([transition[3] for transition in mini_batch])
    dones = np.array([transition[4] for transition in mini_batch])

    target = self.model.predict(states)
    target_next = self.target_model.predict(next_states)

    for i in range(self.batch_size):
        target[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(target_next[i]) * (1 - dones[i])

    self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)


def update_target_model(self):
    self.target_model.set_weights(self.model.get_weights())


def play_game(self):
    game_window = select_game_window()
    state, width, height = get_screen(game_window)
    state_size = (width, height, 3)
    action_size = 4

    self.model = DeepQNetwork(state_size, action_size)
    self.target_model = DeepQNetwork(state_size, action_size)
    self.target_model.set_weights(self.model.get_weights())

    total_episodes = 1000
    steps = 0
    for episode in range(total_episodes):
        done = False
        score = 0
        while not done:
            action = self.get_action(state)
            reward = 0
            if action == 0:
                # Perform left-click action
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                reward = 1
            elif action == 1:
                # Perform right-click action
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                reward = 1
            elif action == 2:
                # Perform move left action
                win32api.keybd_event(win32con.VK_LEFT, 0, 0, 0)
                win32api.keybd_event(win32con.VK_LEFT, 0, win32con.KEYEVENTF_KEYUP, 0)
            elif action == 3:
                # Perform move right action
                win32api.keybd_event(win32con.VK_RIGHT, 0, 0, 0)
                win32api.keybd_event(win32con.VK_RIGHT, 0, win32con.KEYEVENTF_KEYUP, 0)

            next_state, _, _ = get_screen(game_window)
            self.replay_buffer.add([(state, action, reward, next_state, done)])
            state = next_state

            steps += 1
            if steps % 4 == 0:
                self.train()

            if steps % 100 == 0:
                self.update_target_model()

            score += reward

        print("Episode:", episode, " Score:", score)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
