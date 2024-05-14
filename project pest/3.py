import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from keras.models import load_model
from tkinter import *
import tkinter.messagebox
import PIL.Image
import PIL.ImageTk
import time
from tkinter import filedialog

DATADIR = "data"
CATEGORIES = os.listdir(DATADIR)

root = Tk()
root.title("PEST IDENTIFICATION")
root.state('zoomed')
root.configure(bg='#D3D3D3')
root.resizable(width = True, height = True) 
value = StringVar()
panel = Label(root)
model = tf.keras.models.load_model("CNN.model")
def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, 1)
    #img_array = cv2.equalizeHist(img_array)

    #img_array = cv2.Canny(img_array, threshold1=3, threshold2=10)
    img_array = cv2.medianBlur(img_array,1)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array=np.expand_dims(new_array, axis=0)
    return new_array
def detect(filename):
    prediction = model.predict(prepare(filename))
    prediction = list(prediction[0])
    print(prediction)
    l=CATEGORIES[prediction.index(max(prediction))]
    print(CATEGORIES[prediction.index(max(prediction))])
    value.set(CATEGORIES[prediction.index(max(prediction))])


    
    
    
def ClickAction(event=None):
    filename = filedialog.askopenfilename()
    img = PIL.Image.open(filename)
    img = img.resize((250,250))
    img = PIL.ImageTk.PhotoImage(img)
    global panel
    panel = Label(root, image = img)
    panel.image = img
    panel = panel.place(relx=0.435,rely=0.3)
    detect(filename)
    

button = Button(root, text='CHOOSE FILE', font=(None, 18), activeforeground='red', bd=20, bg='cyan', relief=RAISED, height=3, width=20, command=ClickAction)
button = button.place(relx=0.40, rely=0.05)
result = Label(root, textvariable=value, font=(None, 20))
result = result.place(relx=0.465,rely=0.7)
root.mainloop()


`