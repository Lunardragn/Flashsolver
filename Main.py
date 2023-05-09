import tkinter as tk
import cv2
import numpy as np
import time
import pyautogui
import random
import os
import sys
import win32gui
import win32con
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque

HEIGHT = 500
WIDTH = 600
LR = 1e-3
goal_steps = 1000
score_requirement = 50
initial_games = 10000

def get_window_pos(name):
    window = win32gui.FindWindow(None, name)
    if window:
        return win32gui.GetWindowRect(window)
    return None

def take_screenshot(pos):
    x1, y1, x2, y2 = pos
    img = np.array(pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1)))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_resized = cv2.resize(img_gray, (80, 60))
    return img_gray_resized

def press_key(key):
    key = key.lower()
    key_code = ord(key)
    if key == " ":
        key_code = win32con.VK_SPACE
    elif len(key) > 1:
        key_code = win32gui.MapVirtualKey(ord(key), 0)
    win32api.keybd_event(key_code, 0, 0, 0)
    time.sleep(0.01)
    win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(0.01)

def move_mouse(x, y):
    pyautogui.moveTo(x, y)

def click_mouse(button):
    if button == "left":
        pyautogui.click(button="left")
    elif button == "right":
        pyautogui.click(button="right")

def drag_mouse(x, y, button):
    if button == "left":
        pyautogui.dragTo(x, y, button="left")
    elif button == "right":
        pyautogui.dragTo(x, y, button="right")

def scroll_mouse(amount):
    pyautogui.scroll(amount)

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
class Game:
    def __init__(self):
        self.window_name = "Game Window"
        self.game_window_pos = None
        self.game_state = None

    def setup(self):
        self.game_window_pos = get_window_pos(self.window_name)
        self.game_state = take_screenshot(self.game_window_pos)

    def get_game_state(self):
        self.game_state = take_screenshot(self.game_window_pos)
        return self.game_state

    def evaluate_feedback(self, feedback):
        if feedback == "yes":
            # Provide positive reinforcement
            pass
        elif feedback == "no":
            # Provide negative reinforcement
            pass

    def play(self):
        agent = DQNAgent(state_size=self.game_state.shape, action_size=4)
        done = False
        batch_size = 32

        for e in range(initial_games):
            self.setup()
            state = self.get_game_state()
            state = np.reshape(state, (*state.shape, 1))
            for _ in range(goal_steps):
                action = agent.act(state)
                # Perform action based on agent's decision
                # e.g., press_key("w"), move_mouse(x, y), etc.
                next_state = self.get_game_state()
                next_state = np.reshape(next_state, (*next_state.shape, 1))
                reward = 0  # Define the reward based on the game's outcome
                done = False  # Update done based on game's termination condition
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

        # After training, ask for feedback
        feedback = input("Did the agent do well? (yes/no): ")
        self.evaluate_feedback(feedback)
game = Game()
game
