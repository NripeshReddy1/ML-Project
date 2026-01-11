from tkinter import *
import tkinter as tk
import cv2
from tkinter import filedialog
import os
import numpy as np
from PIL import ImageFile                            
import imutils
# global variables
from PIL import ImageTk, Image
global rep
from numpy import load
import skfuzzy as fuzz


from skimage.color import rgb2gray
from tkinter import messagebox
from sklearn.datasets import load_files       
from glob import glob
from keras.preprocessing import image                  
from tkinter import Label


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.config(bg="skyblue")
        
        # changing the title of our master widget
        
        self.master.title("Diabetic retinopathy Detection using Deep learning")
        
        self.pack(fill=BOTH, expand=1)
        
        w = tk.Label(root, 
		 text="Diabetic retinopathy Detection using Deep learning",
		 fg = "black",    
		 bg =  "#FFFFFF",
		 font = "Helvetica 20 bold italic")
        w.pack()
        w.place(x=200, y=0)

        # creating a button instance
        quitButton = Button(self,command=self.load, text="LOAD IMAGE",bg="#FFFF00",fg="#4C0099",activebackground="dark red",width=20)
        quitButton.place(x=50, y=100,anchor="w")
        quitButton = Button(self,command=self.preprocess,text="preprocessing",bg="#FFFF00",fg="#4C0099",activebackground="dark red",width=20)
        quitButton.place(x=50,y=200,anchor="w")
        quitButton = Button(self,command=self.segmentation, text="segment",bg="#FFFF00",fg="#4C0099",activebackground="dark red",width=20)
        quitButton.place(x=50, y=300,anchor="w")
        quitButton = Button(self,command=self.classification,text="PREDICT",bg="#FFFF00",activebackground="dark red",fg="#4C0099",width=20)
        quitButton.place(x=50, y=400,anchor="w")
        
       
        load = Image.open("logo.jfif")
        render = ImageTk.PhotoImage(load)

        image1=Label(self, image=render,borderwidth=15, highlightthickness=5, height=100, width=100)
        image1.image = render
        image1.place(x=400, y=50)

        image2=Label(self, image=render,borderwidth=15, highlightthickness=5, height=100, width=100)
        image2.image = render
        image2.place(x=400, y=250)
        
        
        image4=Label(self, image=render,borderwidth=15, highlightthickness=5, height=100, width=100)
        image4.image = render
        image4.place(x=400, y=450)
        
        image5=Label(self, image=render,borderwidth=15, highlightthickness=5, height=100, width=100)
        image5.image = render
        image5.place(x=650, y=450)
        
        

#       Functions

    def load(self, event=None):
        global rep
        rep = filedialog.askopenfilenames()
        img = cv2.imread(rep[0])
        
        #Input_img=img.copy()
        print(rep[0])
        self.from_array = Image.fromarray(cv2.resize(img,(150,150)))
        load = Image.open(rep[0])
        render = ImageTk.PhotoImage(load.resize((150,150)))
        image1=Label(self, image=render,borderwidth=15, highlightthickness=5, height=100, width=100, bg='pink')
        image1.image = render
        image1.place(x=400, y=50)
        
    def close_window(): 
        Window.destroy()
  
    def preprocess(self, event=None):
        global rep
        img = cv2.imread(rep[0])
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.from_array = Image.fromarray(cv2.resize(hsv_img,(150,150)))
        render = ImageTk.PhotoImage(self.from_array)
        image2=Label(self, image=render,borderwidth=15, highlightthickness=5, height=100, width=100, bg='pink')
        image2.image = render
        image2.place(x=400, y=250)

  
    
    def segmentation(self, event=None):
        from skimage.color import rgb2gray
        from skimage.filters import threshold_otsu
        from skimage import img_as_ubyte
        from skimage.morphology import closing, square
        from skimage.segmentation import clear_border
        from skimage.measure import label, regionprops
        from skimage.color import label2rgb
        from skimage import data, io, filters
        from matplotlib import pyplot as plt
        from skimage import img_as_ubyte
        import numpy as np
        import cv2
        import skfuzzy as fuzz
        from sklearn.cluster import KMeans


        img_org = cv2.imread(rep[0])
        image = cv2.resize(img_org, (150, 150))

        # Fuzzy C-means segmentation
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
           
        # Convert image to fuzzy c-means compatible format (flatten and normalize)
        image_flatten = image_gray.flatten()
        image_flatten = np.expand_dims(image_flatten, axis=1)

        # Apply fuzzy c-means clustering
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        image_flatten.T, c=3, m=2, error=0.005, maxiter=1000)


        # Assigning each pixel to the class with the highest membership value
        cluster_membership = np.argmax(u, axis=0).reshape(image_gray.shape)
        cluster_membership_visual = (cluster_membership / np.max(cluster_membership) * 255).astype(np.uint8)

        # Convert cluster_membership to a PIL Image and resize it
        cluster_membership_img = Image.fromarray(cluster_membership_visual)
        cluster_membership_img_resized = cluster_membership_img.resize((150, 150))

        # Display the image in Tkinter
        render = ImageTk.PhotoImage(cluster_membership_img_resized)
          
        image4=Label(self, image=render,borderwidth=15, highlightthickness=5, height=100, width=100, bg='white')
        image4.image = render
        image4.place(x=400, y=450)

        # K-means clustering segmentation

        # Convert BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the RGB image to a numpy array
        img_array = np.float32(image_rgb.reshape((-1, 3)))

        # Define the number of clusters for K-means segmentation
        num_clusters = 3  # Change the number of clusters as needed

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(img_array)

        # Retrieve labels and centroids from K-means clustering
        labels = kmeans.labels_
        centers = np.uint8(kmeans.cluster_centers_)

        # Replace pixel values with centroid values
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(image_rgb.shape)
        

        # Resize and create ImageTk for K-means segmented image
        #render_kmeans = ImageTk.PhotoImage(segmented_image.resize((150, 150)))


        pil_image = Image.fromarray(segmented_image)

# Resize the PIL Image
        resized_image = pil_image.resize((150, 150), Image.ANTIALIAS)
        render_kmeans = ImageTk.PhotoImage(resized_image)

        
       
        
        image5=Label(self, image=render_kmeans,borderwidth=15, highlightthickness=5, height=100, width=100, bg='white')
        image5.image = render_kmeans
        image5.place(x=650, y=450)


        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    def classification(self, event=None):
        global T,rep
        clas1 = [item[10:-1] for item in sorted(glob("./dataset/*/"))]
        from keras.preprocessing import image                  
        from tqdm import tqdm
        def path_to_tensor(img_path, width=224, height=224):
            print(img_path)
            img = image.load_img(img_path, target_size=(width, height))
            x = image.img_to_array(img)
            return np.expand_dims(x, axis=0)
        def paths_to_tensor(img_paths, width=224, height=224):
            list_of_tensors = [path_to_tensor(img_paths, width, height)]
            return np.vstack(list_of_tensors)
        from tensorflow.keras.models import load_model
        model = load_model('training.h5')
        
       
        test_tensors = paths_to_tensor(rep[0])/255
        pred=model.predict(test_tensors)
        x=np.argmax(pred);
        print('Given image is  = '+clas1[x])
        res='predicted image is '+clas1[x]
        
        T = Text(self, height=5, width=45)
        T.place(x=800, y=600)
        T.insert(END,res)
        

   
    
     

  
                
root = Tk()
root.geometry("1400x720")
app = Window(root)
root.mainloop()

        
