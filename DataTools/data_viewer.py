import os
import tkinter as tk
from tkinter import messagebox
from tkinter import *
from PIL import ImageTk, Image
import pickle 
import DataPoint
import numpy as np
from saveData import * 


win = tk.Tk()
win.geometry('700x700')  # set window size
# win.resizable(0, 0)  # fix window

panel = tk.Label(win)
panel.pack()

filehandler = open('DatasetFinal.obj', 'rb') 
data = pickle.load(filehandler)
filehandler.close()
dataIter = iter(data)
indexer = 0

labels = []
def next_img(event):
    global image, entry, indexer
    # if(event != None):
        # try:
        #     num = int(entry.get())
            
        # except:
        #     entry.delete(0, END)
        #     tk.messagebox.showerror("Error", "Input Error")
        #     return
        # labels.append(num)
        # entry.delete(0, END)

    if indexer < len(data) :
        image = data[indexer]
        indexer += 1
    else:
        win.destroy()
        # newData = saveDataSet(data,labels,5) 
        # print(labels)
        return
    # try:
    #     image = next(dataIter)  # get the next image from the iterator
    # except StopIteration: # No more images, adds the labels and saves dataset
    #     win.destroy()
    #     newData = saveDataSet(data,labels,1) 
    #     print(newData[4].num_of_cubes)
    #     return 
    # load the image and display it
    screen_width = panel.winfo_width()
    img = Image.fromarray(image.image)
    w = img.size[0]
    h = img.size[1]
    img = img.resize((2*w,2*h))
    img = ImageTk.PhotoImage(img)
    panel.img = img  # keep a reference so it's not garbage collected
    panel['image'] = img

def prev_img(e):
    global indexer
    
    if indexer >= 0:
        indexer -= 2
        image = data[indexer]
        # labels.pop()
    screen_width = panel.winfo_width()
    img = Image.fromarray(image.image)
    w = img.size[0]
    h = img.size[1]
    img = img.resize((2*w,2*h))
    img = ImageTk.PhotoImage(img)
    panel.img = img  # keep a reference so it's not garbage collected
    panel['image'] = img
    indexer += 1
    # entry.delete(0, END)

#Create an Entry widget to accept User Input
# entry= Entry(win, width= 25)
# entry.focus_set()
# entry.pack()

#Create an Event for pressing enter
win.bind('<Return>',next_img)
win.bind('<KP_Enter>',next_img)
win.bind('<Left>', prev_img)
win.bind('<Right>', next_img)

# show the first image
next_img(None)

win.mainloop()