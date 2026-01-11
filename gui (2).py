from tkinter import *
import tkinter as tk
import cv2
from tkinter import filedialog, messagebox
import os
import numpy as np
from PIL import ImageTk, Image
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.models import load_model
from glob import glob
global T, rep


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.config(bg="white")
        self.master.title("Roseplant Disease Prediction")
        self.pack(fill=BOTH, expand=1)

        

        # Title label
        w = Label(self, text="Roseplant Disease Prediction", fg="black", bg="#FFFFFF", font="Helvetica 20 bold italic")
        w.pack()
        w.place(x=400, y=0)

        self.image_labels = []  # Initialize the list to hold image labels
        self.create_labels()

        # Button instances
        Button(self, command=self.load, text="LOAD IMAGE", bg="#FFFF00", fg="#4C0099", activebackground="dark red", width=20).place(x=50, y=200, anchor="w")
        Button(self, command=self.classification, text="PREDICT", bg="#FFFF00", fg="#4C0099", activebackground="dark red", width=20).place(x=50, y=400, anchor="w")
        Button(self, command=self.preprocessing, text="PREPROCESSING", bg="#FFFF00", fg="#4C0099", activebackground="dark red", width=20).place(x=50, y=250, anchor="w")
        Button(self, command=self.segmentation, text="SEGMENTATION", bg="#FFFF00", fg="#4C0099", activebackground="dark red", width=20).place(x=50, y=300, anchor="w")
        Button(self, command=self.extract_features, text="EXTRACT FEATURES", bg="#FFFF00", fg="#4C0099", activebackground="dark red", width=20).place(x=50, y=350, anchor="w")
        Button(self, command=self.refresh, text="REFRESH", bg="#FFFF00", fg="#4C0099", activebackground="dark red", width=20).place(x=50, y=450, anchor="w")
##
    def create_labels(self):
        # Create and store labels in the image_labels list
        for i in range(4):  # Assuming you have 4 labels for images
            label = Label(self, borderwidth=15, highlightthickness=5, height=150, width=150)
            label.place(x=(250 + i * 250), y=90)  # Adjust positions
            self.image_labels.append(label)

        # Display a placeholder logo in the labels
        load = Image.open(r"logo.jfif")
        render = ImageTk.PhotoImage(load.resize((200, 200)))
        for label in self.image_labels:
            label.config(image=render)
            label.image = render

    def load(self, event=None):
        global rep
        rep = filedialog.askopenfilenames()
        img = cv2.imread(rep[0])
        self.from_array = Image.fromarray(cv2.resize(img, (200, 200)))
        render = ImageTk.PhotoImage(self.from_array)
        self.image_labels[0].config(image=render)
        self.image_labels[0].image = render

    def preprocessing(self, event=None):
        img = cv2.imread(rep[0])
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.from_array = Image.fromarray(cv2.resize(hsv_img, (200, 200)))
        render = ImageTk.PhotoImage(self.from_array)
        self.image_labels[1].config(image=render)
        self.image_labels[1].image = render

    def segmentation(self, event=None):
        img = cv2.imread(rep[0])
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gauss = cv2.GaussianBlur(hsv_img, (5, 5), 0)

        # KMeans segmentation
        pixel_values = gauss.reshape((-1, 1))
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 4
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()].reshape(gauss.shape)

        self.from_array = Image.fromarray(cv2.resize(segmented_image, (200, 200)))
        render = ImageTk.PhotoImage(self.from_array)
        self.image_labels[2].config(image=render)
        self.image_labels[2].image = render

##        # Dilation (optional, can be shown as a fourth image)
##        kernel = np.ones((5, 5), np.uint8)
##        dilated_image = cv2.dilate(segmented_image, kernel, iterations=1)
##        self.from_array = Image.fromarray(cv2.resize(dilated_image, (200, 200)))
##        render = ImageTk.PhotoImage(self.from_array)


        kernel = np.ones((5, 5), np.uint8)

        # Example bitwise operation: AND operation with a kernel
        bitwise_image = cv2.bitwise_and(hsv_img, segmented_image, mask=None)

        # Resize and convert to PIL image
        self.from_array = Image.fromarray(cv2.resize(bitwise_image, (200, 200)))

        # Update the label with the new image
        render = ImageTk.PhotoImage(self.from_array)
        self.image_labels[3].config(image=render)
        self.image_labels[3].image = render

    def extract_features(self):
        if len(self.from_array.mode) != 1:
            gray_image = self.from_array.convert('L')
        else:
            gray_image = self.from_array

        gray_image = np.array(gray_image)
        distances = [1, 2]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

        contrast = graycoprops(glcm, 'contrast')
        correlation = graycoprops(glcm, 'correlation')
        energy = graycoprops(glcm, 'energy')
        homogeneity = graycoprops(glcm, 'homogeneity')

        # Display features
        features = {
            'Contrast': contrast,
            'Correlation': correlation,
            'Energy': energy,
            'Homogeneity': homogeneity
        }

        m = "\n".join([f"{name}: {value}" for name, value in features.items()])

        self.result_text1 = Text(self, height=15, width=60,background= "white")
        self.result_text1.place(x=550, y=300)

        self.result_text1.delete(1.0, END)
        self.result_text1.insert(END, m)

    def classification(self, event=None):
        
        global T, rep



   
        from glob import glob
        #from keras.preprocessing import image
        from tensorflow.keras.models import load_model

        # Get the list of classes from the directory structure
        clas1 = [item[10:-1] for item in sorted(glob("./dataset/*/"))]
        print(clas1)
        from tensorflow.keras.preprocessing import image                  
        #from tqdm import tqdm
        def path_to_tensor(img_path, width=224, height=224):
            print(img_path)
            img = image.load_img(img_path, target_size=(width, height))
            x = image.img_to_array(img)
            return np.expand_dims(x, axis=0)
        def paths_to_tensor(img_paths, width=224, height=224):
            list_of_tensors = [path_to_tensor(img_paths, width, height)]
            return np.vstack(list_of_tensors)
        from tensorflow.keras.models import load_model

        # Load the pre-trained model
        model1 = load_model('trained_model_DNN1.h5')

        # Load and preprocess the test image
        test_tensors1 = paths_to_tensor(rep[0]) / 255

        # Make predictions using the model
        pred1 = model1.predict(test_tensors1)
        print(pred1)
        # Get the index of the predicted class
        predicted_class_index1 = np.argmax(pred1)
        print(predicted_class_index1)

        # Display the predicted class
        res3 = 'Predicted disease is ' + clas1[predicted_class_index1]


        # Display the predicted class in the GUI
##        T = Text(self, height=5, width=40)
##        T.place(x=550, y=600)
##        T.insert(END, res3)
##        self.result_text2 = Text(self, height=15, width=40,background= "white")
##        self.result_text2.place(x=550, y=300)
##        self.result_text2.delete(1.0, END)
##        self.result_text2.insert(END, res3)




        if predicted_class_index1 == 0:
            # Path to your file
            yt_file_path = './files/Cache.txt'

            # Read the content of the file
            with open(yt_file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()

            # Create Tkinter window
            root = tk.Tk()
            root.geometry("700x700")  # Set window size

            # Create Text widget with specified height and width
            T = tk.Text(root, bg="#ffdab9", height=35, width=150, bd=5, relief='sunken')  # Adjust height and width as needed
            T.place(x=30, y=50)  # Set position

            # Create a tag with specified font size
            T.tag_configure("font_size", font=("Helvetica", 16),justify='center')  # Change "Helvetica" and 12 to desired font and size

            # Insert the content of the file into the Text widget
            T.insert(tk.END, file_content.strip())

            # Specify the line number you want to change (0-based index)
            line_number = 0  # Change to your desired line number (e.g., 2 for the third line)

            # Get the start and end indices for the specified line
            start_index = f"{line_number + 1}.0"  # line_number + 1 because of 0-based index
            end_index = f"{line_number + 1}.end"

            # Apply the tag to that specific line
            T.tag_add("font_size", start_index, end_index)




        if predicted_class_index1 == 1:
            # Path to your file
            yt_file_path = './files/DownyMildew.txt'

            # Read the content of the file
            with open(yt_file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()

            # Create Tkinter window
            root = tk.Tk()
            root.geometry("700x700")  # Set window size

            # Create Text widget with specified height and width
            T = tk.Text(root, bg="#ffdab9", height=35, width=150, bd=5, relief='sunken')  # Adjust height and width as needed
            T.place(x=30, y=50)  # Set position

            # Create a tag with specified font size
            T.tag_configure("font_size", font=("Helvetica", 16), justify='center')  # Change "Helvetica" and 12 to desired font and size

            # Insert the content of the file into the Text widget
            T.insert(tk.END, file_content.strip())

            # Specify the line number you want to change (0-based index)
            line_number = 0  # Change to your desired line number (e.g., 2 for the third line)

            # Get the start and end indices for the specified line
            start_index = f"{line_number + 1}.0"  # line_number + 1 because of 0-based index
            end_index = f"{line_number + 1}.end"

            # Apply the tag to that specific line
            T.tag_add("font_size", start_index, end_index)


        if predicted_class_index1 == 2:
            # Path to your file
            yt_file_path = './files/blackspot.txt'

            # Read the content of the file
            with open(yt_file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()

            # Create Tkinter window
            root = tk.Tk()
            root.geometry("700x700")  # Set window size

            # Create Text widget with specified height and width
            T = tk.Text(root, bg="#ffdab9", height=35, width=150, bd=5, relief='sunken')  # Adjust height and width as needed
            T.place(x=30, y=50)  # Set position

            # Create a tag with specified font size
            T.tag_configure("font_size", font=("Helvetica", 16), justify='center')  # Change "Helvetica" and 12 to desired font and size

            # Insert the content of the file into the Text widget
            T.insert(tk.END, file_content.strip())

            # Specify the line number you want to change (0-based index)
            line_number = 0  # Change to your desired line number (e.g., 2 for the third line)

            # Get the start and end indices for the specified line
            start_index = f"{line_number + 1}.0"  # line_number + 1 because of 0-based index
            end_index = f"{line_number + 1}.end"

            # Apply the tag to that specific line
            T.tag_add("font_size", start_index, end_index)



        if predicted_class_index1 == 3:
            # Path to your file
            yt_file_path = './files/botrytisblight.txt'

            # Read the content of the file
            with open(yt_file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()

            # Create Tkinter window
            root = tk.Tk()
            root.geometry("700x700")  # Set window size

            # Create Text widget with specified height and width
            T = tk.Text(root, bg="#ffdab9", height=35, width=150, bd=5, relief='sunken')  # Adjust height and width as needed
            T.place(x=30, y=50)  # Set position

            # Create a tag with specified font size
            T.tag_configure("font_size", font=("Helvetica", 16), justify='center')  # Change "Helvetica" and 12 to desired font and size

            # Insert the content of the file into the Text widget
            T.insert(tk.END, file_content.strip())

            # Specify the line number you want to change (0-based index)
            line_number = 0  # Change to your desired line number (e.g., 2 for the third line)

            # Get the start and end indices for the specified line
            start_index = f"{line_number + 1}.0"  # line_number + 1 because of 0-based index
            end_index = f"{line_number + 1}.end"

            # Apply the tag to that specific line
            T.tag_add("font_size", start_index, end_index)


        if predicted_class_index1 == 4:
            # Path to your file
            yt_file_path = './files/powdery mildrew.txt'

            # Read the content of the file
            with open(yt_file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()

            # Create Tkinter window
            root = tk.Tk()
            root.geometry("700x700")  # Set window size

            # Create Text widget with specified height and width
            T = tk.Text(root, bg="#ffdab9", height=35, width=150, bd=5, relief='sunken')  # Adjust height and width as needed
            T.place(x=30, y=50)  # Set position

            # Create a tag with specified font size
            T.tag_configure("font_size", font=("Helvetica", 16), justify='center')  # Change "Helvetica" and 12 to desired font and size

            # Insert the content of the file into the Text widget
            T.insert(tk.END, file_content.strip())

            # Specify the line number you want to change (0-based index)
            line_number = 0  # Change to your desired line number (e.g., 2 for the third line)

            # Get the start and end indices for the specified line
            start_index = f"{line_number + 1}.0"  # line_number + 1 because of 0-based index
            end_index = f"{line_number + 1}.end"

            # Apply the tag to that specific line
            T.tag_add("font_size", start_index, end_index)   


       
    def refresh(self):
        for label in self.image_labels:
            label.config(image='')
            label.image = None

        self.result_text1.delete(1.0, END)
        #self.after(2,self.clear_text)
        
        self.result_text2.delete(1.0, END)
        #self.result_text.delete(1.0, END)
        global rep
        rep = None

class LoginWindow(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.config(bg='white')
        self.master.title("Login")

        bg_image = Image.open("rose.jpg")
        bg_render = ImageTk.PhotoImage(bg_image)
        self.background_label = Label(self, image=bg_render)
        self.background_label.image = bg_render
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.pack(fill=BOTH, expand=1)
        text_label = Label(self, text="Welcome to the Roseplant Disease Prediction", bg='white', fg='black', font=("Helvetica", 16))
        text_label.place(x=450, y=200)  # Adjust the position as needed



        

##        self.login_button = Button(self, text="chatbot", command=self.login)
##        self.login_button.place(x=650, y=200)

        self.login_button = Button(self, text="upload", command=self.login)
        self.login_button.place(x=650, y=300)

    def login(self):
        self.master.switch_frame(Window)



class MainApplication(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.title("Roseplane disease Prediction")
        self.geometry("1400x720")
        self.current_frame = None
        self.switch_frame(LoginWindow)
        
    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = new_frame
        self.current_frame.pack()

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()        

##root = Tk()
##root.geometry("1400x720")
##app = Window(root)
##root.mainloop()




'''


from tkinter import *
import tkinter as tk
import cv2
from tkinter import filedialog, messagebox
import os
import numpy as np
from PIL import ImageTk, Image
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from glob import glob
from keras.preprocessing import image

class Window(Frame):
    def __init__(self, master=None):
        super().__init__(master)                 
        self.master = master
        self.config(bg="white")
        self.master.title("Roseplant Disease Prediction")
        self.pack(fill=BOTH, expand=1)

        # Initialize instance variables
        self.rep = None
        self.model = load_model('trained_model_DNN.h5')
        self.image_labels = []  
        
        # Title label
        Label(self, text="Roseplant Disease Prediction", fg="black", bg="#FFFFFF", font="Helvetica 20 bold italic").pack(pady=(10, 0))

        self.create_labels()
        self.create_buttons()

    def create_labels(self):
        for i in range(4):  
            label = Label(self, borderwidth=15, highlightthickness=5, height=150, width=150)
            label.place(x=(250 + i * 250), y=90)
            self.image_labels.append(label)

        load = Image.open("logo.jfif")
        render = ImageTk.PhotoImage(load.resize((200, 200)))
        for label in self.image_labels:
            label.config(image=render)
            label.image = render

    def create_buttons(self):
        buttons = [
            ("LOAD IMAGE", self.load),
            ("PREPROCESSING", self.preprocessing),
            ("SEGMENTATION", self.segmentation),
            ("EXTRACT FEATURES", self.extract_features),
            ("PREDICT", self.classification),
            ("REFRESH", self.refresh)
        ]
        for i, (text, command) in enumerate(buttons):
            Button(self, command=command, text=text, bg="#FFFF00", fg="#4C0099", activebackground="dark red", width=20).place(x=50, y=200 + i * 50, anchor="w")

    def load(self):
        self.rep = filedialog.askopenfilenames()
        if not self.rep:
            return
        img = cv2.imread(self.rep[0])
        if img is None:
            messagebox.showerror("Error", "Could not load image.")
            return
        self.display_image(img, self.image_labels[0])

    def display_image(self, img, label):
        resized_img = cv2.resize(img, (200, 200))
        self.from_array = Image.fromarray(resized_img)
        render = ImageTk.PhotoImage(self.from_array)
        label.config(image=render)
        label.image = render

    def preprocessing(self):
        img = cv2.imread(self.rep[0])
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.display_image(hsv_img, self.image_labels[1])

    def segmentation(self):
        img = cv2.imread(self.rep[0])
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(hsv_img, (5, 5), 0)

        pixel_values = gauss.reshape((-1, 1))
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 4
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()].reshape(gauss.shape)

        self.display_image(segmented_image, self.image_labels[2])

        kernel = np.ones((5, 5), np.uint8)
        dilated_image = cv2.dilate(segmented_image, kernel, iterations=1)
        self.display_image(dilated_image, self.image_labels[3])

    def extract_features(self):
        gray_image = self.from_array.convert('L') if self.from_array.mode != 'L' else self.from_array
        gray_image = np.array(gray_image)
        
        distances = [1, 2]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

        features = {
            'Contrast': graycoprops(glcm, 'contrast'),
            'Correlation': graycoprops(glcm, 'correlation'),
            'Energy': graycoprops(glcm, 'energy'),
            'Homogeneity': graycoprops(glcm, 'homogeneity')
        }

        self.result_text1 = Text(self, height=15, width=60, background="white")
        self.result_text1.place(x=550, y=300)
        self.result_text1.delete(1.0, END)
        self.result_text1.insert(END, "\n".join([f"{name}: {value}" for name, value in features.items()]))

    def classification(self):
        if self.rep is None:
            messagebox.showwarning("Warning", "No image loaded for classification.")
            return

        test_tensors1 = self.prepare_image_for_prediction(self.rep[0])
        pred1 = self.model.predict(test_tensors1)
        predicted_class_index1 = np.argmax(pred1)
        self.display_prediction(predicted_class_index1)

    def prepare_image_for_prediction(self, img_path):
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        return np.expand_dims(x, axis=0) / 255

    def display_prediction(self, predicted_class_index):
        class_names = [item[10:-1] for item in sorted(glob("./dataset/*/"))]
        result = f'Predicted disease is: {class_names[predicted_class_index]}'
        messagebox.showinfo("Prediction Result", result)

    def refresh(self):
        for label in self.image_labels:
            label.config(image='')
            label.image = None
        self.result_text1.delete(1.0, END)
        self.rep = None


class LoginWindow(Frame):
    def __init__(self, master=None):
        super().__init__(master)                 
        self.master = master
        self.config(bg='white')
        self.master.title("Login")

        bg_image = Image.open("rose.jpg")
        bg_render = ImageTk.PhotoImage(bg_image)
        self.background_label = Label(self, image=bg_render)
        self.background_label.image = bg_render
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.pack(fill=BOTH, expand=1)
        
        text_label = Label(self, text="Welcome to the Roseplant Disease Prediction", bg='white', fg='black', font=("Helvetica", 16))
        text_label.place(x=450, y=200)

        self.login_button = Button(self, text="upload", command=self.login)
        self.login_button.place(x=650, y=300)

    def login(self):
        self.master.switch_frame(Window)


class MainApplication(Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Roseplant Disease Prediction")
        self.geometry("1400x720")
        self.current_frame = None
        self.switch_frame(LoginWindow)

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = new_frame
        self.current_frame.pack()


if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
'''

