#create a button
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

# create a button
root = Tk()


def browsefunc1():
    modelpath = filedialog.askopenfilename()
    pathlabel1.config(text=modelpath)


browsebutton1 = Button(root, text="Browse", command=browsefunc1)
browsebutton1.grid(row=0, column=2)

pathlabel1 = Label(root)
pathlabel1.grid(row=0, column=1)


# e1 = Entry(pathlabel1).place(x=100,y=80)

def browsefunc2():
    imagepath = filedialog.askopenfilename()
    pathlabel2.config(text=imagepath)


browsebutton2 = Button(root, text="Browse", command=browsefunc2)
browsebutton2.grid(row=1, column=2)

pathlabel2 = Label(root)
pathlabel2.grid(row=1, column=1)
# e2 = Entry(pathlabel2).grid(x=100,y=80)

# create checkbox
var1 = IntVar()
Checkbutton(root, text='flag1', variable=var1).grid(row=2, column=2, sticky=W)
var2 = IntVar()
Checkbutton(root, text='flag2', variable=var2).grid(row=3, column=2, sticky=W)
var3 = IntVar()
Checkbutton(root, text='flag3', variable=var3).grid(row=4, column=2, sticky=W)


def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename
def open_img():
    x = openfn()
    img = Image.open(x)
    # img = img.resize((200, 200), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.grid(row=1, column=4)
btn = Button(root, text='Display image', command=open_img).grid(row=0, column=4)
# create process button
processbtn = Button(root, text="Process!").grid(row=6, column=1)

#def quantify_image(image, bins=(4, 6, 3)):
	# compute a 3D color histogram over the image and normalize it
	#hist = cv2.calcHist([image], [0, 1, 2], None, bins,
		#[0, 180, 0, 256, 0, 256])
	#hist = cv2.normalize(hist, hist).flatten()
	# return the histogram
	#return hist
def image_process():
    modelpath = filedialog.askopenfilename()
    pathlabel1.config(text=modelpath)
    imagepath = filedialog.askopenfilename()
    pathlabel2.config(text=imagepath)

processbtn = Button(root, text="Process!",command=image_process).grid(row=6, column=1)

confidenceThreshold = 0.5
NMSThreshold = 0.3

modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3.weights'

labelsPath = 'coco.names'
labels = open(labelsPath).read().strip().split('\n')

np.random.seed(10)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

image = cv2.imread('dog.jpg')
(H, W) = image.shape[:2]

#Determine output layer names
layerName = net.getLayerNames()
layerName = [layerName[i[0] - 1] for i in net.getUnconnectedOutLayers()]

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)
net.setInput(blob)
layersOutputs = net.forward(layerName)

boxes = []
confidences = []
classIDs = []

for output in layersOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > confidenceThreshold:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY,  width, height) = box.astype('int')
            x = int(centerX - (width/2))
            y = int(centerY - (height/2))

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

#Apply Non Maxima Suppression
detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)

if(len(detectionNMS) > 0):
    for i in detectionNMS.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#histr = cv2.calcHist([image],[0],None,[256],[0,256])
#hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#features = quantify_image(hsv, bins=(3, 3, 3))
cv2.imshow('Image', image)
#cv2.waitKey(0)
mainloop()