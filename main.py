import tkinter as tk
from tkinter import ttk
import random as rand

n = 3


# Function to update the debug label with the sum of the slider values
def update_debug_label():
    total = sum([value.get() for value in vector_values])
    debug_label.config(text=f"Sum: {total:.2f}")


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
        command=lambda value: update_debug_label(),
        from_=-1,
        to=1,
        variable=current_value
    )
    current_slider.pack()
    vector_values.append(current_value)
    sliders.append(current_slider)

# Set up some debugging help preemptively
debug_label = ttk.Label(window, text="Sum: 0.00")
debug_label.pack()
update_debug_label()

# Display
window.mainloop()
