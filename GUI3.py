import tkinter as tk
from tkinter import filedialog, messagebox
from djitellopy import Tello
import cv2
import threading
import torch
from PIL import Image, ImageTk

# Load the YOLOv5 model (replace with your actual model path)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Project\Project\yolov5\runs\train\exp3\weights\best.pt')

# Initialize the Tello drone
tello = Tello()

# Global variable to control video detection loop
stop_detection = False

def connect_drone():
    tello.connect()
    battery = tello.get_battery()
    messagebox.showinfo("Tello Battery", f"Battery level: {battery}%")

def fly_drone():
    tello.takeoff()
    tello.move_forward(200)
    tello.rotate_clockwise(180)
    tello.move_forward(200)
    tello.land()

def start_fly_drone():
    drone_thread = threading.Thread(target=fly_drone)
    drone_thread.start()

def video_feed():
    tello.streamon()
    cap = tello.get_frame_read()
    
    while not stop_detection:
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
        
    tello.streamoff()

def start_video_detection():
    global stop_detection
    stop_detection = False
    video_thread = threading.Thread(target=video_feed)
    video_thread.start()

def stop_video_detection():
    global stop_detection
    stop_detection = True

def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
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
            break
    
    cap.release()

def upload_and_process_video():
    global stop_detection
    stop_detection = False
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if file_path:
        video_thread = threading.Thread(target=process_video, args=(file_path,))
        video_thread.start()

# Create the main window
root = tk.Tk()
root.title("Tello Drone Controller")
root.geometry("1000x600")  # Set window size
root.configure(bg='#f0f0f0')  # Set background color

# Create a heading label
heading_label = tk.Label(root, text="Drowning Detection System", font=('Helvetica', 18, 'bold'), bg='#f0f0f0')
heading_label.pack(pady=10)

# Create frames for buttons and video feed
button_frame = tk.Frame(root, bg='#f0f0f0')
button_frame.pack(side='left', fill='y', padx=10, pady=10)

video_frame = tk.Frame(root, bg='#000000')
video_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)

# Create and place the buttons with styles
button_style = {
    'bg': '#4CAF50',
    'fg': 'white',
    'font': ('Helvetica', 12, 'bold'),
    'width': 25,
    'height': 2,
    'relief': 'raised',
    'bd': 5
}

connect_button = tk.Button(button_frame, text="Connect Drone", command=connect_drone, **button_style)
connect_button.pack(pady=10)

fly_button = tk.Button(button_frame, text="Start and Fly Drone", command=start_fly_drone, **button_style)
fly_button.pack(pady=10)

start_video_button = tk.Button(button_frame, text="Start Video Detection", command=start_video_detection, **button_style)
start_video_button.pack(pady=10)

stop_video_button = tk.Button(button_frame, text="Stop Video Detection", command=stop_video_detection, **button_style)
stop_video_button.pack(pady=10)

upload_video_button = tk.Button(button_frame, text="Upload and Process Video", command=upload_and_process_video, **button_style)
upload_video_button.pack(pady=10)

# Video feed display
video_label = tk.Label(video_frame, bg='#000000')
video_label.pack(fill='both', expand=True)

# Run the main loop
root.mainloop()
