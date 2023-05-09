import psutil
import pyautogui
import torch
import cv2
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

# initialize PyTorch model
model = torch.nn.Sequential(
    torch.nn.Linear(8, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 3),
    torch.nn.Sigmoid()
)

# initialize OpenCV object detector
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# initialize GUI
root = tk.Tk()
canvas = tk.Canvas(root, width=600, height=400)
canvas.pack()

# function to get screenshot of game
def get_screenshot():
    # get list of running processes
    processes = []
    for process in psutil.process_iter(['pid', 'name']):
        if process.info['name'] == 'game.exe':
            processes.append(process)
    
    # get process with highest memory usage
    process = max(processes, key=lambda p: p.memory_info().rss)
    
    # get screenshot of game window
    window = pyautogui.getWindowsWithTitle(process.info['name'])[0]
    x, y, width, height = window.left, window.top, window.width, window.height
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return np.array(screenshot)

# function to preprocess image for model input
def preprocess(image):
    # convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # resize to 64x64
    image = cv2.resize(image, (64, 64))
    # convert to float32 and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    # reshape to (1, 1, 64, 64)
    image = np.reshape(image, (1, 1, 64, 64))
    return torch.from_numpy(image)

# function to get model output and execute action
def execute_action(image):
    # preprocess image
    input = preprocess(image)
    # get model output
    output = model(input)
    # get action from output
    action = torch.argmax(output).item()
    # execute action
    if action == 0:
        # move mouse to center of screen
        pyautogui.moveTo(pyautogui.size().width / 2, pyautogui.size().height / 2)
    elif action == 1:
        # left click
        pyautogui.click()
    elif action == 2:
        # press up arrow key
        pyautogui.press('up')

# function to update GUI
def update_gui():
    # get screenshot
    screenshot = get_screenshot()
    # detect objects in screenshot
    objects = detector.detectMultiScale(screenshot, scaleFactor=1.1, minNeighbors=5)
    # draw boxes around detected objects
    for (x, y, w, h) in objects:
        cv2.rectangle(screenshot, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # convert screenshot to ImageTk format
    image = Image.fromarray(screenshot)
    image_tk = ImageTk.PhotoImage(image)
    # update canvas
    canvas.create_image(0, 0, anchor='nw', image=image_tk)
    # schedule next GUI update
    root.after(100, update_gui)
root.mainloop()
