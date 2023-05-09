import psutil
import pyautogui
import torch
import cv2
import tkinter as tk

# Function to find game process using psutil
def find_process(process_name):
    for process in psutil.process_iter():
        if process.name().lower() == process_name.lower():
            return process.pid
    return None

# Function to create the GUI
def create_gui():
    root = tk.Tk()
    root.title("Deep-Q Neural Network Game Player")

    # Create label to show status
    status_label = tk.Label(root, text="Status: Not running")
    status_label.pack()

    # Create label to show object recognition
    object_label = tk.Label(root, text="Object recognition: ")
    object_label.pack()

    # Create label to show current action
    action_label = tk.Label(root, text="Current action: ")
    action_label.pack()

    # Create button to show OpenCV what to look for
    learn_button = tk.Button(root, text="Learn Object", command=learn_object)
    learn_button.pack()

    # Create button to select program to play
    select_button = tk.Button(root, text="Select Program", command=select_program)
    select_button.pack()

    root.mainloop()

# Function to teach the neural network using deep-Q learning
def deep_q_learning():
    # Load game
    game_pid = find_process("game.exe")
    game_window = pyautogui.getWindowsWithTitle("Game Window")[0]
    game_window.activate()

    # Initialize neural network using pytorch
    net = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 3)
    )

    # Loop to play game and teach the neural network
    while True:
        # Take screenshot and use OpenCV to recognize objects
        screenshot = pyautogui.screenshot()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        # object_recognition = opencv_function(screenshot)

        # Use neural network to determine action
        action = net(screenshot)
        # execute_action(action)

        # Check for positive reinforcement (progress to next level)
        if check_progression():
            # reward_network(action)
            break

    # Save trained neural network
    torch.save(net.state_dict(), "game_network.pth")

# Function to manually select the program to play
def select_program():
    # Use tkinter file dialog to select program
    file_path = tk.filedialog.askopenfilename(initialdir="/", title="Select Program", filetypes=(("Exe files", "*.exe"), ("All files", "*.*")))
    # launch_game(file_path)

# Function to teach OpenCV what to look for
def learn_object():
    # Use tkinter file dialog to select object image
    file_path = tk.filedialog.askopenfilename(initialdir="/", title="Select Object Image", filetypes=(("Image files", "*.png;*.jpg"), ("All files", "*.*")))
    # object_image = opencv_function(file_path)
    # save_object_image(object_image)

# Main function to start the program
def main():
    create_gui()
    deep_q_learning()

if __name__ == "__main__":
    main()
