import tkinter as tk
from tkinter import messagebox
from djitellopy import Tello
import cv2
import threading
import time
import torch

# Load the YOLOv5 model (replace 'your_model.pt' with your actual model)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Project\Project\yolov5\runs\train\exp\weights\best.pt')

# Initialize the Tello drone
tello = Tello()

def connect_drone():
    tello.connect()
    battery = tello.get_battery()
    messagebox.showinfo("Tello Battery", f"Battery level: {battery}%")

def fly_drone():
    tello.takeoff()
    tello.move_forward(500)
    tello.rotate_clockwise(90)
    tello.move_forward(500)
    tello.rotate_clockwise(90)
    tello.move_forward(500)
    tello.rotate_clockwise(90)
    tello.move_forward(500)
    tello.rotate_clockwise(90)
    tello.land()

def start_fly_drone():
    drone_thread = threading.Thread(target=fly_drone)
    drone_thread.start()

def video_feed():
    tello.streamon()
    cap = tello.get_frame_read()
    
    while True:
        frame = cap.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLOv5 model
        results = model(frame)
        
        # Display results
        result_img = results.render()[0]
        img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        
        # Convert image to PhotoImage and display in GUI
        img = cv2.resize(img, (640, 480))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        video_label.config(image=img_tk)
        video_label.image = img_tk
        
        if stop_detection:
            tello.streamoff()
            break
    
    tello.streamoff()

def start_video_detection():
    global stop_detection
    stop_detection = False
    video_thread = threading.Thread(target=video_feed)
    video_thread.start()

def stop_video_detection():
    global stop_detection
    stop_detection = True

# Create the main window
root = tk.Tk()
root.title("Tello Drone Controller")

# Create and place the buttons
connect_button = tk.Button(root, text="Connect Drone", command=connect_drone)
connect_button.pack(pady=10)

fly_button = tk.Button(root, text="Start and Fly Drone", command=start_fly_drone)
fly_button.pack(pady=10)

start_video_button = tk.Button(root, text="Start Video Detection", command=start_video_detection)
start_video_button.pack(pady=10)

stop_video_button = tk.Button(root, text="Stop Video Detection", command=stop_video_detection)
stop_video_button.pack(pady=10)

# Video feed display
video_label = tk.Label(root)
video_label.pack()

# Run the main loop
root.mainloop()
