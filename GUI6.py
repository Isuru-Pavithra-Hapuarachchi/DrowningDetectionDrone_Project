import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from djitellopy import Tello
import cv2
import threading
import torch
from PIL import Image, ImageTk
import requests

# Load the YOLOv5 model (replace with your actual model path)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Project\Project\yolov5\runs\train\exp5\weights\best.pt')

# Initialize the Tello drone
tello = Tello()

# Global variable to control video detection loop
stop_detection = False

# Telegram bot credentials
bot_token = '6607213240:AAHuQcPCW-fKTwU0_bxI6saxHriV-7aMW4c'
chat_id = '5971179299'

# Counter for consecutive detections
detection_counter = 0

# Set the drone speed (valid range is 10 to 100 cm/s)
drone_speed = 50  # Example speed value

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=data)
    return response

def add_new_telegram_user():
    global chat_id
    new_chat_id = simpledialog.askstring("Input", "Enter new Telegram Chat ID:", parent=root)
    if new_chat_id:
        chat_id = new_chat_id
        messagebox.showinfo("Success", f"New Telegram Chat ID set to: {chat_id}")

def connect_drone():
    tello.connect()
    battery = tello.get_battery()
    messagebox.showinfo("Tello Battery", f"Battery level: {battery}%")

def disconnect_drone():
    tello.end()
    messagebox.showinfo("Tello Disconnect", "Drone disconnected")

def fly_drone():
    global stop_detection
    tello.set_speed(drone_speed)  # Set the speed of the drone
    tello.takeoff()
    tello.move_up(100)  # Ascend 1 meter

    # Fly in a predefined path: left, right, left
    tello.move_left(500) 
    tello.rotate_clockwise(360)  # Rotate 360 degrees
    tello.move_right(500) 
    tello.rotate_clockwise(360)  
    tello.move_left(500) 
    tello.rotate_clockwise(360)  
    tello.move_right(500) 
    tello.rotate_clockwise(360) 
    tello.move_left(500) 
    tello.rotate_clockwise(360)  
    tello.move_right(500) 
    tello.land()

def start_fly_drone():
    drone_thread = threading.Thread(target=fly_drone)
    drone_thread.start()

def video_feed():
    global detection_counter, stop_detection
    tello.streamon()
    cap = tello.get_frame_read()
    
    while not stop_detection:
        frame = cap.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLOv5 model
        results = model(frame)
        
        # Check if any objects of class "2" (drowning) are detected
        detections = results.xyxy[0]
        if any(detection[5] == 2 for detection in detections):
            detection_counter += 1
        else:
            detection_counter = 0
        
        if detection_counter >= 5:
            # send_telegram_message("Drowning person detected in live feed!")  # Enable notification sending
            detection_counter = 0
            stop_detection = True
            tello.land()  # Land the drone upon detection
        
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
    global stop_detection, detection_counter
    stop_detection = False
    detection_counter = 0
    video_thread = threading.Thread(target=video_feed)
    video_thread.start()

def stop_video_detection():
    global stop_detection
    stop_detection = True

def process_video(file_path):
    global detection_counter
    cap = cv2.VideoCapture(file_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLOv5 model
        results = model(frame)
        
        # Check if any objects of class "2" (drowning) are detected
        detections = results.xyxy[0]
        if any(detection[5] == 2 for detection in detections):
            detection_counter += 1
        else:
            detection_counter = 0
        
        if detection_counter >= 5:
            # send_telegram_message("Drowning person detected in video!")  # Enable notification sending
            detection_counter = 0
        
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
    global stop_detection, detection_counter
    stop_detection = False
    detection_counter = 0
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if file_path:
        video_thread = threading.Thread(target=process_video, args=(file_path,))
        video_thread.start()

def emergency_land():
    global stop_detection
    stop_detection = True
    tello.land()
    messagebox.showinfo("Emergency Land", "Drone is landing due to emergency.")

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

add_user_button = tk.Button(button_frame, text="Add Telegram User", command=add_new_telegram_user, **button_style)
add_user_button.pack(pady=10)

disconnect_button = tk.Button(button_frame, text="Disconnect Drone", command=disconnect_drone, **button_style)
disconnect_button.pack(pady=10)

emergency_button = tk.Button(button_frame, text="Emergency Land", command=emergency_land, **button_style)
emergency_button.pack(pady=10)

# Video feed display
video_label = tk.Label(video_frame, bg='#000000')
video_label.pack(fill='both', expand=True)

# Run the main loop
root.mainloop()
