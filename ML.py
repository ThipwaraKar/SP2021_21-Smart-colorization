import tkinter as tk #For GUI
import os #clear file, open dialogue, find list/path folder
import cv2
import sys
import numpy as np
import random
from PIL import ImageTk, Image
from tkinter import filedialog
from model import NeuralNetwork
import Local_Binary_Patterns as lbp

def Feature_extraction():

    count = 0
    for i in range(8):
        for r, d, f in os.walk('Training\\' + str(i)):
            for file in f:
                if file.endswith(".bmp"):
                    count += 1

    X_train = np.zeros(shape=(count, 26))
    y_train = np.zeros(shape=(count, 8))

    count = 0
    for i in range(8):
        for r, d, f in os.walk('Training\\' + str(i)):
            for file in f:
                if file.endswith(".bmp"):
                    file = 'Training\\' + str(i) + '\\' + file

                    gray = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                    #gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # LBP
                    cls_lbp = lbp.LocalBinaryPatterns(24, 8)
                    feature = cls_lbp.describe(binary)

                    c = 0
                    for m in range(26):
                        X_train[count, c] = feature[m]
                        c += 1

                    target = np.zeros(shape=(1, 8))
                    target[0, i] = 1

                    for k in range(8):
                        y_train[count, k] = target[0, k]

                    count += 1


    return X_train, y_train

def ANN_training():

    [X_train, y_train] = Feature_extraction()

    # Neural network training
    input_size = 26
    hidden_size = 24
    output_size = 15
    epoch = 200000
    mse = 0.0001
    layer_sizes = [input_size, hidden_size, output_size]
    nn = NeuralNetwork(epoch, mse)
    nn.create(layer_sizes)
    nn.train(X_train, y_train)

    # evaluate on train data
    train_accuracy = nn.evaluate(X_train, y_train)
    print("Train accuracy: ", "{0:.2f}%".format(train_accuracy * 100))

    # save model
    model_path = "nn_model.xml"
    nn.save_model(model_path)

def Classify():

    # Clear old files
    filelist = [f for f in os.listdir('Classify') if f.endswith(".bmp")]
    for f in filelist:
        os.remove(os.path.join('Classify', f))

    # Opendialog
    file = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select file",
                                      filetypes=(("PNG files", "*.png"), ("all files", "*.*")))
    rgb = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    # cv2.imshow('All Contours', rgb[:, :, 3])
    gray = rgb[:, :, 3];

    # Binary (OTSU)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # contour
    # get the contours in the thresholded image
    rgb_fill_color = rgb.copy()
    (cnts, _) = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, cnts, -1, (0, 255, 0), 3)
    cv2.imwrite('contour.bmp', rgb)
    rgb_contour = cv2.resize(rgb.copy(), (1024, 768), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('All Contours', rgb_contour)

    # fill color
    #cv2.drawContours(rgb_fill_color, cnts, -1, (50, 140, 255), thickness=-1)
    #cv2.imwrite('filled.bmp', rgb_fill_color)

    # Check position of interested
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite('binary_test.bmp', binary)

    cs = 0
    for cnt in contours:

        (x, y, w, h) = cv2.boundingRect(cnt)

        ROI = binary[y:y + h, x:x + w]

        cv2.imwrite('Classify/' + str(cs) + '.bmp', ROI)

        # Feature extraction ( LBP )
        cls_lbp = lbp.LocalBinaryPatterns(24, 8)
        feature = cls_lbp.describe(ROI)
        # print(feature)

        # Neural network classification
        X_test = np.zeros(shape=(1, 26))

        c = 0
        for m in range(26):
            X_test[0, c] = feature[m]
            c += 1

        # NN Classify
        nn = NeuralNetwork()
        nn.load_model('nn_model.xml')
        result = nn.predict(X_test)
        print(result)
        cv2.putText(rgb, str(result).replace('[', '').replace(']', ''), (x + int(w / 2), y + int(h / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0, 255), 5, lineType=cv2.LINE_AA)

        cs += 1

        #fill color by class
        result = int(result)
        color_name = ''
        if result == 0:
            color_name = 'Body.txt'
        if result == 1:
            color_name = 'Eye.txt'
        elif result == 2:
            color_name = 'Body.txt'
        elif result == 3:
            color_name = 'Mouth.txt'
        elif result == 4:
            color_name = 'Hair.txt'
        elif result == 5:
            color_name = 'Body.txt'
        elif result == 6:
            color_name = 'Hair.txt'
        elif result == 7:
            color_name = 'Body.txt'

        #condition that filled same color on the body parts [head, ears, throat, and can't define]
        #if result == 0 or result == 2 or result == 5 or result == 7:

         #   num_line =
        #else:
        num_line = random.randint(0, 4)

        print('Color\\' + color_name)
        with open('Color\\' + color_name) as file:
            c = 0
            for line in file:
                if c == num_line:
                    R = int(line.split(',')[0].strip())
                    G = int(line.split(',')[1].strip())
                    B = int(line.split(',')[2].strip())
                    print('{} , {} , {}'.format(R, G, B))
                    cv2.drawContours(rgb_fill_color, [cnt], -1, (B, G, R), thickness=-1)

                    #save as .jpg type but if want to change just change .jpg to .png or .bmp
                    cv2.imwrite('filled.jpg', rgb_fill_color)
                    break
                c += 1

    # Resize
    # r, g, b, a = rgb.split()
    # img = Image.merge("RGB", (r, g, b))

    rgb_extract = cv2.resize(rgb, (800, 500), interpolation=cv2.INTER_LINEAR)
    # cv2.imshow('x', rgb_extract)

    # Show in tkinter label
    prevImg = Image.fromarray(rgb_extract)
    imgtk = ImageTk.PhotoImage(image=prevImg)
    panel.imgtk = imgtk
    panel.configure(image=imgtk)

def Extract_UV():
    # Clear old files
    filelist = [f for f in os.listdir('Extract') if f.endswith(".bmp")]
    for f in filelist:
        os.remove(os.path.join('Extract', f))

    # Opendialog
    file = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select file",
                                      filetypes=(("PNG files", "*.png"), ("all files", "*.*")))
    rgb = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    gray = rgb[:, :, 3];

    # cv2.imwrite('rgb.bmp', rgb)

    # Gray scale
    # gray = rgb #cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Binary (OTSU)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Closing
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite('binary.bmp', binary)

    # Blob
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = 0
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if (w * h) > 10:
            ROI = gray[y:y + h, x:x + w]
            save_name = 'Extract/' + str(c) + '.bmp'
            cv2.imwrite(save_name, ROI)
            c += 1

if __name__ == "__main__":

    # GUI
    global mainWindow, w, panel

    mainWindow = tk.Tk()
    mainWindow.title('Auto color fill UV mapping')
    mainWindow.resizable(width=False, height=False)

    window_height = 500
    window_width = 800

    screen_width = mainWindow.winfo_screenwidth()
    screen_height = mainWindow.winfo_screenheight()

    x_cordinate = int((screen_width / 2) - (window_width / 2))
    y_cordinate = int((screen_height / 2) - (window_height / 2))

    mainWindow.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

    panel = tk.Label(mainWindow, compound=tk.CENTER, anchor=tk.CENTER, bg="white")
    panel.place(x=10, y=10, bordermode="outside", height=480, width=640)

    button_webcam_start = tk.Button(mainWindow, text="Classify", command=Classify, height=3, width=20)
    button_webcam_start.place(x=660, y=10, width=130, height=40)

    button_webcam_stop = tk.Button(mainWindow, text="Training ML", command=ANN_training, height=3, width=20)
    button_webcam_stop.place(x=660, y=55, width=130, height=40)

    button_webcam_video = tk.Button(mainWindow, text="Extract UV map", command=Extract_UV, height=3, width=20)
    button_webcam_video.place(x=660, y=100, width=130, height=40)

    button_webcam_exit = tk.Button(mainWindow, text="Exit", command=exit, height=3, width=20)
    button_webcam_exit.place(x=660, y=450, width=130, height=40)

    mainWindow.mainloop()