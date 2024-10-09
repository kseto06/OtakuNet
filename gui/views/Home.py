import customtkinter as ctk
from PIL import Image, ImageTk

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Prompt import prompt_view


def home_view(frame):
    # Configure to hold multiple frames
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    # Home Screen Frame
    home_frame = ctk.CTkFrame(frame) #Define the home screen frame as a frame of the global frame
    home_frame.grid(row=0, column=0, sticky="nsew")

    # Set the background NN image
    set_bg(home_frame)

    # Title
    title = ctk.CTkLabel(master=home_frame, text="OtakuNet", font=("Open Sans", 60, "bold"), fg_color="transparent") 
    title.grid(row=0, column=0, padx=20, pady=100, sticky="n")

    # Button
    button = ctk.CTkButton(master=home_frame, text="Start Recommendations", font=("Open Sans", 16), width=250, height=100, fg_color="transparent", command=lambda: prompt_view(frame))
    button.grid(row=0, column=0, padx=515, pady=450)

    frame.tkraise()
    

def set_bg(frame):
    #Load bg image
    image = Image.open('./views/images/neuralnetwork.jpeg')
    photo = ImageTk.PhotoImage(image) #Configure to tk

    # Create CTkLabel to hold image
    bg_label = ctk.CTkLabel(frame, image=photo, text="")
    bg_image = photo
    bg_label.place(relwidth=1, relheight=1)