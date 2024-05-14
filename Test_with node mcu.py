import tkinter as tk
from tkinter import filedialog
import cv2
import os
import numpy as np
from keras.models import load_model
from PIL import Image, ImageTk
import threading

# Function to load the model
def load_cnn_model():
    return load_model('CNN.model')

data_dir = "data"
class_names = os.listdir(data_dir)

# Global variable to indicate when to perform analysis
perform_analysis = False

def capture_video():
    global perform_analysis
    # Initialize the webcam
    cam = cv2.VideoCapture(1)
    while True:
        ret, frame = cam.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            
            # Update the label with new frame
            video_label.imgtk = img  # Keep a reference
            video_label.config(image=img)
            
            if perform_analysis:
                perform_analysis = False
                analyze_frame(frame)
        else:
            print("Failed to capture video from webcam")

def analyze_frame(frame):
    try:
        resized_frame = cv2.resize(frame, (100, 100))
        resized_frame = np.expand_dims(resized_frame, axis=0)
        resized_frame = resized_frame / 255.0
        
        prediction = model.predict(resized_frame)
        predicted_class = np.argmax(prediction)
        
        result_label.config(text=f"Predicted Class: {class_names[predicted_class]}")
    except Exception as e:
        print(f"Error during frame analysis: {e}")

def on_q_pressed(event):
    global perform_analysis
    perform_analysis = True

# Create the Tkinter window
root = tk.Tk()
root.title("Live Video Analysis")

# Label to display the live video
video_label = tk.Label(root)
video_label.pack()

# Label to display the analysis result
result_label = tk.Label(root, text="")
result_label.pack()

# Load the CNN model
model = load_cnn_model()

# Bind the 'q' key to the on_q_pressed function
root.bind('q', on_q_pressed)

# Create a thread for capturing video
video_thread = threading.Thread(target=capture_video)
video_thread.start()

# Start the Tkinter main loop
root.mainloop()
