import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter.messagebox

# Declare global variables for frames
global left_frame
global right_frame
global image
global canvas
global original_image_displayed
global filtered_image_displayed

# Create a fixed-size canvas
canvas_width = 1600
canvas_height = 600

# Define a function to apply image filters
def apply_filter(filter_type):
    global image
    img_copy = image.copy()  # Create a copy of the image

    if filter_type == "Blur":
        kernel_size = (5, 5)  # Define the kernel size for blurring
        filtered_image = cv2.blur(img_copy, kernel_size)
    elif filter_type == "Sharpen":
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])  # Define the sharpening kernel
        filtered_image = cv2.filter2D(img_copy, -1, kernel)
    elif filter_type == "Edge Detection":
        if image is not None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply the Sobel operator for edge detection
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
            magnitude = cv2.magnitude(sobel_x, sobel_y)

            # Convert the magnitude to CV_8U for display
            magnitude = cv2.convertScaleAbs(magnitude)

            # Display the detected edges on the interface
            display_filtered_image(magnitude)
        else:
            print("No image to perform edge detection. Please open an image first.")
    elif filter_type == "Image Enhancement":
        # Apply adaptive histogram equalization for image enhancement
        if image is not None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply adaptive histogram equalization to enhance image contrast
            enhanced_image = exposure.equalize_adapthist(gray_image, clip_limit=0.03)
            # Convert the enhanced image to BGR format for display
            filtered_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
            display_filtered_image(filtered_image)
        else:
            print("No image to enhance. Please open an image first.")
    elif filter_type == "Image Segmentation":
        if image is not None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, segmented_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            display_filtered_image(segmented_image)
        else:
            print("No image to perform segmentation. Please open an image first.")
    elif filter_type == "Object Recognition":
        # Use Haar Cascade Classifier for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw on img_copy
        filtered_image = img_copy

    # Display the processed image on the interface
    display_filtered_image(filtered_image)

# Define a function to apply image distortion
def apply_distortion(distortion_type):
    global image
    global distorted_image

    if distortion_type == "Translation":
        translation_matrix = np.float32([[1, 0, 50], [0, 1, 50]])
        distorted_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    elif distortion_type == "Rotation":
        rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 45, 1)
        distorted_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    elif distortion_type == "Scaling":
        def set_scale_factor():
            try:
                factor = float(scale_factor_entry.get())
                new_width = int(image.shape[1] * factor)
                new_height = int(image.shape[0] * factor)
                distorted_image = cv2.resize(image, (new_width, new_height))
                display_filtered_image(distorted_image)
                scale_window.destroy()
            except ValueError:
                tk.messagebox.showerror("Error", "Please enter a valid number!")

        scale_window = tk.Toplevel(window)
        scale_window.title("Enter Scaling Factor")

        tk.Label(scale_window, text="Enter scaling factor (e.g., 2 for double size, 0.5 for half size):").pack(pady=10)
        scale_factor_entry = tk.Entry(scale_window)
        scale_factor_entry.pack(pady=10)

        tk.Button(scale_window, text="OK", command=set_scale_factor).pack(pady=10)
    elif distortion_type == "Shear":
        shear_matrix = np.float32([[1, 0.3, 0], [0.3, 1, 0]])
        distorted_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))
    else:
        distorted_image = image

    display_filtered_image(distorted_image)

# Define a function to display the original image on the interface
def display_original_image(img):
    global canvas
    global original_image_displayed
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((canvas_width // 2, canvas_height), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    original_image_displayed = img  # Save as a global variable to prevent garbage collection

def display_filtered_image(img):
    global canvas
    global filtered_image_displayed

    if img.shape[1] > canvas_width // 2 or img.shape[0] > canvas_height:
        img = cv2.resize(img, (canvas_width // 2, canvas_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)
    canvas.create_image(canvas_width // 2, 0, anchor=tk.NW, image=img)
    filtered_image_displayed = img  # Save as a global variable to prevent garbage collection

# Create the main window
window = tk.Tk()
window.title("Image Processing Application")

# Create the menu bar
menu_bar = tk.Menu(window)
window.config(menu=menu_bar)

# Create the File menu
file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)

# Add the Open Image option
def open_image():
    global image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        image = cv2.imread(file_path)
        display_original_image(image)

file_menu.add_command(label="Open Image", command=open_image)

# Create the Image Processing menu
image_processing_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Image Processing", menu=image_processing_menu)

# Add filter type options
image_processing_menu.add_command(label="Blur", command=lambda: apply_filter("Blur"))
image_processing_menu.add_command(label="Sharpen", command=lambda: apply_filter("Sharpen"))
image_processing_menu.add_command(label="Edge Detection", command=lambda: apply_filter("Edge Detection"))
image_processing_menu.add_command(label="Image Segmentation", command=lambda: apply_filter("Image Segmentation"))
image_processing_menu.add_command(label="Image Enhancement", command=lambda: apply_filter("Image Enhancement"))

# Add object recognition option
image_processing_menu.add_command(label="Object Recognition", command=lambda: apply_filter("Object Recognition"))

# Create the Image Distortion menu
distortion_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Image Distortion", menu=distortion_menu)

# Add image distortion type options
distortion_menu.add_command(label="Translation", command=lambda: apply_distortion("Translation"))
distortion_menu.add_command(label="Rotation", command=lambda: apply_distortion("Rotation"))
distortion_menu.add_command(label="Scaling", command=lambda: apply_distortion("Scaling"))
distortion_menu.add_command(label="Shear", command=lambda: apply_distortion("Shear"))

# Create a fixed-size canvas
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height)
canvas.pack()

# Run the main loop
window.mainloop()
