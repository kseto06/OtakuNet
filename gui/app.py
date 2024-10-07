import customtkinter as ctk
from views import Home 

ctk.set_appearance_mode("dark-blue")
ctk.set_default_color_theme("dark-blue")

# Root (main window)
root = ctk.CTk()
root.geometry("1400x800")
root.title("OtakuNet - The Anime Content-Based Filtering App")
root.resizable(False, False) #Prevent resizing

# Global Frame
frame = ctk.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

# Configure to hold multiple frames
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)

# Call the home_view function to create the home screen and add the button to switch to the recommend screen
Home.home_view(frame)

# Execution
frame.tkraise()
root.mainloop()

'''
TODO:
- Sliders for genres, get values from sliders
- Use predict defined in framework/model to get the predictions
- For making new user predictions, we can basically just follow the ipynb demo that I've already created
'''