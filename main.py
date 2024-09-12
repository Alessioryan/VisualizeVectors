import tkinter as tk
from tkinter import *
import random as rand
from tkinter import ttk
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageTk
from models import Autoencoder
import numpy as np

n = 3

# Load in the model in eval mode
f = torch.load('Models/linear_3dim.pth')
f.eval()


# Function to update the debug label with the sum of the slider values
def update_image():
    vector = torch.Tensor([value.get() for value in vector_values])
    print(vector)
    image = np.maximum(f.decoder(vector).view(28, 28).detach().numpy() * 255, 0)
    print(image)
    image_pil = Image.fromarray(image.astype(np.uint8) ).resize((200, 200), Image.NEAREST)
    img_updated = ImageTk.PhotoImage(image_pil)
    # Update the canvas image and keep a reference
    canvas.itemconfig(image_on_canvas, image=img_updated)
    canvas.image = img_updated


# Window
window = tk.Tk()
window.geometry("300x300")
window.title('main')

# There are n dimensions to the vector
# Sliders
vector_values = []
sliders = []
for i in range(n):
    current_value = tk.DoubleVar(value=rand.uniform(-1, 1))
    current_slider = ttk.Scale(
        window,
        command=lambda value: update_image(),
        from_=-1,
        to=1,
        variable=current_value
    )
    current_slider.pack()
    vector_values.append(current_value)
    sliders.append(current_slider)

# Image
canvas = Canvas(window, width=300, height=300)
canvas.pack()
img = ImageTk.PhotoImage(Image.open("Reconstructions/linear_3dim_5epochs.png"))
image_on_canvas = canvas.create_image(50, 50, anchor=NW, image=img)

# Set up some debugging help preemptively
update_image()

# Display
window.mainloop()
