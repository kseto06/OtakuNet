import customtkinter as ctk
import pandas as pd
#from Home import home_view
from functions.predictions import prediction
from PIL import Image
import requests
from io import BytesIO
import random

def load_img_from_url(url: str):
    response = requests.get(url)
    img_data = response.content
    img = Image.open(BytesIO(img_data))
    return img

def back(frame): 
    from Prompt import prompt_view
    prompt_view(frame)

def recommend_view(frame, values: list):
    # Configure to hold multiple frames
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    # Recommend Frame
    recommend_frame = ctk.CTkFrame(frame)
    recommend_frame.grid(row=0, column=0, sticky="nsew")

    # Title
    recommend_label = ctk.CTkLabel(master=recommend_frame, text="  Recommendations", font=("Open Sans", 40, "bold"), fg_color="transparent")
    recommend_label.place(x=470, y=10)

    # Button
    button = ctk.CTkButton(master=recommend_frame, text="Back", font=("Open Sans", 15), fg_color="transparent", border_color="white", border_width=1, width=100, height=60, command=lambda: back(frame))
    button.place(x=20, y=10)

    # Compute the prediction with the values
    preds = prediction(values)
    preds.pop(0) # Remove the first index which contiains the headers

    # Display the first 4 predictions, randomized from the prediction array:

    num_preds = min(4, len(preds)) #Get the min of 4 displayed or the length of the predictions if the number of preds is < 4
    random.seed(int(random.randrange(1, 100)))
    random_preds = random.sample(preds, num_preds) # Generate a random sample each time

    for i in range(num_preds): 
        if (random_preds[i]): #Check if the predicted index exists
            try:
                # Extract the image of the anime
                # Load the image from the URL
                pil_image = load_img_from_url(random_preds[i][3])

                # Convert it to CTkImage
                ctk_image = ctk.CTkImage(pil_image, size=(120, 120))

                # Create a label to display the image
                image_label = ctk.CTkLabel(master=recommend_frame, text='', image=ctk_image)
                image_label.place(x=1000, y=(i+1)*145)

            except:
                # Catch header exception, which prevents the 0th index headers from being displayed (already popped however so this should not trigger). 
                # It occurs on the image URL step since no URL is associated with it
                print("Invalid anime")
                continue

            # Create ctk label with the name index
            label = ctk.CTkLabel(master=recommend_frame, text=str(random_preds[i][1]), font=("Open Sans", 20)) 
            label.place(x=100, y=20+(i+1)*145)

            # Show genres tied to the anime
            genres_label = ctk.CTkLabel(master=recommend_frame, text='Genres: '+str(random_preds[i][2]), font=("Open Sans", 10))
            genres_label.place(x=100, y=50+(i+1)*145) 