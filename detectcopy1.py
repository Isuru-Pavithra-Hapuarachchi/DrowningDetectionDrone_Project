import argparse
import cv2
import torch
from djitellopy import Tello

def main(opt):
    print(f"Loading YOLOv5 model from {opt.weights}...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=opt.weights)
    print("Model loaded successfully.")

    # Set model parameters
    model.conf = opt.conf_thres  # Confidence threshold
    model.iou = 0.45  # NMS IoU threshold (you can adjust as needed)
    model.classes = None  # (optional list) filter by class

    # Check the source type
    if opt.source == 'tello':
        print("Connecting to Tello drone...")
        tello = Tello()
        tello.connect()
        print("Connected to Tello drone.")
        tello.streamon()
        print("Tello video stream started.")
        cap = tello.get_frame_read()
    else:
        print(f"Opening video source: {opt.source}")
        cap = cv2.VideoCapture(opt.source)
        if not cap.isOpened():
            print(f"Failed to open video source: {opt.source}")
            return

    print("Starting video processing...")
    while True:
        if opt.source == 'tello':
            frame = cap.frame
        else:
            ret, frame = cap.read()
            if not ret:
                print("No more frames to read or failed to read frame.")
                break

        # Convert the frame to a format suitable for YOLOv5
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = model(img, size=opt.img_size)

        # Render the results on the frame
        results.render()

        # Display the frame
        cv2.imshow("YOLOv5 Stream", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    if opt.source == 'tello':
        tello.streamoff()
        print("Tello video stream stopped.")
    else:
        cap.release()
    cv2.destroyAllWindows()
    print("Video processing stopped, resources released.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/exp/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam, or 'tello' for Tello drone
    opt = parser.parse_args()
    main(opt)
