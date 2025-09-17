from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import filedialog
from keras.models import model_from_json, load_model
from random import randrange
from numpy.random import randn
import cv2
from matplotlib import pyplot

main = tkinter.Tk()
main.title("Diabetic Retinopathy Image Synthesis with Generative Adversarial Network")
main.geometry("1300x1200")

global filename
global gan_model
global predict_model

def upload():
    text.delete('1.0', END)
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END, filename + " loaded\n")

def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    print(x_input.shape)
    return x_input

def create_plot(examples, n):
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(examples[i, :, :])
    pyplot.show()

def ganModel():
    global gan_model
    gan_model = load_model('c:/Users/mdkou/OneDrive/Desktop/Diabetic Retinopathy Image Synthesis with Generative Adversarial Network/model/generator_model_080.h5')
    latent_points = generate_latent_points(200, 200)
    X = gan_model.predict(latent_points)
    text.insert(END, 'GAN model generated\n')
    text.insert(END, 'GAN generated new images size : ' + str(X.shape) + "\n\n")
    create_plot(X, 10)

def predictModel():
    global predict_model
    text.delete('1.0', END)
    with open('model/train.json', "r") as json_file:
        loaded_model_json = json_file.read()
        predict_model = model_from_json(loaded_model_json)

    predict_model.load_weights("model/train.h5")
    predict_model._make_predict_function()
    print(predict_model.summary())
    text.insert(END, 'See black console to view model summary')

def getPrediction(img):
    result = 'none'
    img1 = np.asarray(img)
    img1 = img1.reshape(1, 32, 32, 3)  # Ensure this matches the model's input shape
    preds = predict_model.predict(img1)  # Predicting class of image severity
    predict = np.argmax(preds)  # Get the class value
    result = 'none'
    if predict == 0:   
        result = 'No DR'
    elif predict == 1:   
        result = 'Mild'
    elif predict == 2:   
        result = 'Moderate'
    elif predict == 3:   
        result = 'Severe'
    elif predict == 4:   
        result = 'Proliferative DR'
    return result    

def predictSeverity():
    if not gan_model:
        text.insert(END, 'Please load GAN model first.\n')
        return
    if not predict_model:
        text.insert(END, 'Please load prediction model first.\n')
        return
    
    latent_points = generate_latent_points(200, 200) 
    X = gan_model.predict(latent_points)
    for i in range(10):  # Displaying 10 images for simplicity
        index = randrange(200) 
        img = X[index, :, :] 
        result = getPrediction(img)
        img = cv2.resize(img, (300, 300)) 
        cv2.putText(img, 'Prediction Result: ' + result, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Image ID: ' + str(index) + ' Prediction Result: ' + result, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # Ensure windows are properly closed

def predict():
    if not predict_model:
        text.insert(END, 'Please load the prediction model first.\n')
        return

    file_path = filedialog.askopenfilename(initialdir=".", title="Select Fundus Image",
                                           filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if not file_path:
        text.insert(END, "No image selected.\n")
        return

    # Load image using OpenCV
    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, (32, 32))  # Match model input size
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Get prediction result
    result = getPrediction(img_rgb)

    # Resize original image for display and add prediction label
    display_img = cv2.resize(img, (400, 400))
    cv2.putText(display_img, 'Prediction: ' + result, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show image with prediction result
    cv2.imshow("Prediction Result", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Log prediction in the text widget
    text.insert(END, "Selected Image: " + file_path + "\n")
    text.insert(END, "Predicted Diabetic Retinopathy Severity: " + result + "\n\n")

def closeApp():
    main.destroy()
    
font = ('times', 16, 'bold')
title = Label(main, text='Diabetic Retinopathy Image Synthesis with Generative Adversarial Network')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=100)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=480, y=100)
text.config(font=font1)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Fundus Dataset", command=upload)
uploadButton.place(x=50, y=100)
uploadButton.config(font=font1)

ganButton = Button(main, text="Load GAN Model", command=ganModel)
ganButton.place(x=50, y=150)
ganButton.config(font=font1)

modelButton = Button(main, text="Load Diabetic Retinopathy Prediction Model", command=predictModel)
modelButton.place(x=50, y=200)
modelButton.config(font=font1)

predictButton = Button(main, text="Generate GAN Image & Predict Severity", command=predictSeverity)
predictButton.place(x=50, y=250)
predictButton.config(font=font1)

predictButton = Button(main, text="Predict", command=predict)
predictButton.place(x=50, y=300)
predictButton.config(font=font1)


closeButton = Button(main, text="Exit", command=closeApp)
closeButton.place(x=50, y=350)
closeButton.config(font=font1)

main.config(bg='OliveDrab2')
main.mainloop()
