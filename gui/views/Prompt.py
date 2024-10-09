import customtkinter as ctk

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Recommend import recommend_view

def prompt_view(frame):
    # Define genre list
    genres = ['Action', 'Award Winning', 'Sci-Fi', 'Adventure', 'Drama',
       'Mystery', 'Supernatural', 'Fantasy', 'Sports', 'Comedy',
       'Romance', 'Slice of Life', 'Suspense', 'Gourmet',
       'Avant Garde', 'Horror', 'Girls Love', 'Boys Love'] 

    # Configure to hold multiple frames
    frame.grid_rowconfigure((0), weight=1)
    frame.grid_columnconfigure((0, 1, 2), weight=1)
    frame.grid_propagate(0)

    # Prompt Frame
    prompt_frame = ctk.CTkFrame(frame)
    prompt_frame.grid(row=0, column=0, sticky="nsew")

    # Title
    prompt_label = ctk.CTkLabel(master=prompt_frame, text="Rate Genres from 0-100", font=("Open Sans", 30, "bold"), fg_color="transparent")
    prompt_label.place(x=490, y=10)

    # Input Boxes
    genre_inputs = generate_inputs(prompt_frame, genres)

    # Button
    button = ctk.CTkButton(master=prompt_frame, text="Recommend", font=("Open Sans", 18), fg_color="transparent", border_color="white", border_width=1, command=lambda: submit(frame, genre_inputs))
    button.place(x=590, y=700)

    frame.tkraise()

# Function to generate inputs dynamically
def generate_inputs(prompt_frame, genres: list, genre_inputs={}): 
    for index, genre in enumerate(genres):
        # For each genre, create a label
        genre_label = ctk.CTkLabel(master=prompt_frame, text=str(genre), font=("Open Sans", 10), height=10)
        genre_label.place(x=415, y=40+index*35)

        # Create input for each genre
        genre_entry = ctk.CTkEntry(master=prompt_frame, placeholder_text='0', width=500, height=20)
        genre_entry.place(x=415, y=60+index*35)
        
        # Store the genre_inputs, which allows us to dynamically extract values of each genre from their respective CTkInputs
        genre_inputs[genre] = genre_entry

    return genre_inputs


# Function to get the values from the inputs
def get_values(genre_inputs):
    values = []
    # Store the values for model to predict
    for genre, entry in genre_inputs.items():
        value = entry.get()
        # Replace unrated as '0':
        if value == '':
            value = '0'

        # Ensure range [0, 100]
        if int(value) < 0:
            value = '0'
        elif int(value) > 100:
            value = '100'

        values.append(int(value)) #Convert from str -> int

    print(values)
    return values

# Function to collect values and switch to next view
def submit(frame, genre_inputs):
    values = get_values(genre_inputs)
    recommend_view(frame, values)