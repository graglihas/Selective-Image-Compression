import tkinter as tk
from PIL import Image, ImageTk
from tkinter import Button, filedialog, Text, END

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
from skimage import data
from skimage.color import rgb2gray
from sklearn.preprocessing import StandardScaler
from skimage.feature import blob_log
from sklearn.cluster import DBSCAN
from io import BytesIO
from shapely.geometry import Polygon
from shapely.geometry import Point


def change_pic():
    global vlabel
    global current_image
    
    root.fileName  = filedialog.askopenfilename(filetypes = (  ( "jpg files", "*.jpg") ,  ( "png files", "*.png") , ( "python files", "*.py")   , ("All files", "*.*") ) )

    loaded_img = Image.open(root.fileName )
    resized_loaded_img = loaded_img.resize((1000, 600),Image.ANTIALIAS)

    root.photo1 = ImageTk.PhotoImage(resized_loaded_img)

    vlabel.configure(image=root.photo1)
    print ("updated" , root.fileName)
    current_image = root.fileName
    print("current_image is " , current_image )

def algo_execution():

	print("Compression Successful")

def clahe100():
    file_name = current_image

    img = cv2.imread(file_name)
    # im=img
    # window = cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Original", 512,512)
    #cv2.imshow("Original", img)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    gray = img[:,:,1]
    # window = cv2.namedWindow("BlackAndWhite", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("BlackAndWhite", 512,512)
    #cv2.imshow("BlackAndWhite", gray)


    median = cv2.bilateralFilter(gray,5, 500, 10)
    # window = cv2.namedWindow("MedianFiltered", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("MedianFiltered", 512,512)
    #cv2.imshow("MedianFiltered", median)


    clahe = cv2.createCLAHE(clipLimit=100, tileGridSize=(5,5))
    cl1 = clahe.apply(median)
    cl1 = cv2.resize(cl1,(512,512))
    plt.imshow(cl1)
    plt.title("AfterCLAHE-100")
    plt.show()
    # cv2.waitKey(0)
    return

##########################################################
def thresholding235():
    file_name = current_image

    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = img[:, :, 1]

    median = cv2.bilateralFilter(gray, 5, 500, 10)

    clahe = cv2.createCLAHE(clipLimit=100, tileGridSize=(5, 5))
    cl1 = clahe.apply(median)
    cl1 = cv2.resize(cl1, (512, 512))

    image = cv2.bilateralFilter(cl1, 5, 500, 10)

    ret, th1 = cv2.threshold(image, 235, 255, cv2.THRESH_BINARY)

    plt.imshow(th1)
    plt.title("AfterThreshold-235")
    plt.show()

##########################################################
def detection():

    file_name = current_image

    image = cv2.imread(file_name)
    image = cv2.resize(image, (512, 512))
    image_gray = rgb2gray(image)

    blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_list = [blobs_log]
    colors = ['yellow']
    titles = ['Laplacian of Gaussian']
    sequence = zip(blobs_list, colors, titles)
    ax = plt.axes()

    rm, xm, ym = 0, 0, 0
    for idx, (blobs, color, title) in enumerate(sequence):
        ax.set_title(title)
        ax.imshow(image, interpolation='nearest')
        for blob in blobs:
            y, x, r = blob
            if (r >= rm) and (70 < y < 442) and (70 < x < 442):
                rm, xm, ym = r, x, y
        c = plt.Circle((xm, ym), rm, color=color, linewidth=2, fill=False)
        ax.add_patch(c)
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()

##########################################################

def removal():
    file_name = current_image

    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = img[:, :, 1]

    median = cv2.bilateralFilter(gray, 5, 500, 10)

    clahe = cv2.createCLAHE(clipLimit=100, tileGridSize=(5, 5))
    cl1 = clahe.apply(median)
    cl1 = cv2.resize(cl1, (512, 512))

    image = cv2.bilateralFilter(cl1, 5, 500, 10)

    ret, th1 = cv2.threshold(image, 235, 255, cv2.THRESH_BINARY)

    image = cv2.imread(file_name)
    image = cv2.resize(image, (512, 512))
    image_gray = rgb2gray(image)

    blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_list = [blobs_log]
    colors = ['yellow']
    titles = ['Laplacian of Gaussian']
    sequence = zip(blobs_list, colors, titles)

    rm, xm, ym = 0, 0, 0
    for idx, (blobs, color, title) in enumerate(sequence):
        for blob in blobs:
            y, x, r = blob
            if (r >= rm) and (70 < y < 442) and (70 < x < 442):
                rm, xm, ym = r, x, y

    # print(ym, xm, rm)

    xc, yc, r = xm, ym, rm  # location and size of the circle
    H, W = th1.shape  # size of the image
    x, y = np.meshgrid(np.arange(W), np.arange(H))  # x and y coordinates per every pixel of the image
    d2 = (x - xc) ** 2 + (y - yc) ** 2  # squared distance from the center of the circle
    mask = d2 < r ** 2  # mask is True inside of the circle

    th1[mask] = 0

    plt.imshow(th1)
    plt.title("OpticDiskRemoved.jpg")
    plt.show()

##########################################################
def clusters():

    file_name = current_image

    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = img[:, :, 1]

    median = cv2.bilateralFilter(gray, 5, 500, 10)

    clahe = cv2.createCLAHE(clipLimit=100, tileGridSize=(5, 5))
    cl1 = clahe.apply(median)
    cl1 = cv2.resize(cl1, (512, 512))

    image = cv2.bilateralFilter(cl1, 5, 500, 10)

    ret, th1 = cv2.threshold(image, 235, 255, cv2.THRESH_BINARY)

    image = cv2.imread(file_name)
    image = cv2.resize(image, (512, 512))
    image_gray = rgb2gray(image)

    blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_list = [blobs_log]
    colors = ['yellow']
    titles = ['Laplacian of Gaussian']
    sequence = zip(blobs_list, colors, titles)

    rm, xm, ym = 0, 0, 0
    for idx, (blobs, color, title) in enumerate(sequence):
        for blob in blobs:
            y, x, r = blob
            if (r >= rm) and (70 < y < 442) and (70 < x < 442):
                rm, xm, ym = r, x, y

    # print(ym, xm, rm)

    xc, yc, r = xm, ym, rm  # location and size of the circle
    H, W = th1.shape  # size of the image
    x, y = np.meshgrid(np.arange(W), np.arange(H))  # x and y coordinates per every pixel of the image
    d2 = (x - xc) ** 2 + (y - yc) ** 2  # squared distance from the center of the circle
    mask = d2 < r ** 2  # mask is True inside of the circle

    th1[mask] = 0

    image_gray = rgb2gray(th1)
    # finding blobs
    blobs_log = blob_log(image_gray, max_sigma=10, num_sigma=10, threshold=.1)
    # radiusof blobs
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    # list containing centers of blobs
    data = []
    for blob in blobs_log:
        discard = []
        y, x, r = blob
        y = int(y)
        x = int(x)
        discard.append(x)
        discard.append(y)
        data.append(discard)
    data_r = data
    data = StandardScaler().fit_transform(data)
    # DBSCAN density based algorithm
    db = DBSCAN(eps=0.3, min_samples=5).fit(data)
    # creating a nparray with number of points set to false
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # setting true to indices of points that are core points
    core_samples_mask[db.core_sample_indices_] = True
    # labelling point to respective clusters including noise point
    labels = db.labels_

    # unique labels is number of cluster + 1(for noise is counted as a cluster)
    unique_labels = set(labels)
    # defining colors for each label

    AllRegions = []
    for k in unique_labels:
        if k == -1:
            continue
        discard = []
        b = 0
        t = 512
        r = 0
        l = 512
        for j in range(0, len(labels)):
            if labels[j] == k:
                if data_r[j][0] > r:
                    r = data_r[j][0]
                if data_r[j][0] < l:
                    l = data_r[j][0]
                if data_r[j][1] < t:
                    t = data_r[j][1]
                if data_r[j][1] > b:
                    b = data_r[j][1]
        cv2.circle(image, (l - 5, t - 5), 10, color=255, thickness=2)
        cv2.circle(image, (l - 5, b + 5), 10, color=255, thickness=2)
        cv2.circle(image, (r + 5, b + 5), 10, color=255, thickness=2)
        cv2.circle(image, (r + 5, t - 5), 10, color=255, thickness=2)
        discard.append([t - 5, l - 5])
        discard.append([b + 5, l - 5])
        discard.append([b + 5, r + 5])
        discard.append([t - 5, r + 5])
        AllRegions.append(discard)

    plt.imshow(image)
    plt.title("Clustered")
    plt.show()
##########################################################
def finalcompressed():


    file_name = current_image

    img = cv2.imread(file_name)
    temp_img = cv2.resize(img, (512, 512))
    temp_file_name1 = file_name.split('/')[-1].split('.')[0]

    cv2.imwrite('images/' + temp_file_name1 + '_orig.jpg' , temp_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # temp_img = cv2.resize(img, (512, 512))
    # cv2.imwrite("images/"+ file_name.split('/')[-1] , temp_img)
    gray = img[:, :, 1]

    median = cv2.bilateralFilter(gray, 5, 500, 10)

    clahe = cv2.createCLAHE(clipLimit=100, tileGridSize=(5, 5))
    cl1 = clahe.apply(median)
    cl1 = cv2.resize(cl1, (512, 512))

    image = cv2.bilateralFilter(cl1, 5, 500, 10)

    ret, th1 = cv2.threshold(image, 235, 255, cv2.THRESH_BINARY)

    image = cv2.imread(file_name)
    image = cv2.resize(image, (512, 512))
    image_gray = rgb2gray(image)

    blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_list = [blobs_log]
    colors = ['yellow']
    titles = ['Laplacian of Gaussian']
    sequence = zip(blobs_list, colors, titles)

    rm, xm, ym = 0, 0, 0
    for idx, (blobs, color, title) in enumerate(sequence):
        for blob in blobs:
            y, x, r = blob
            if (r >= rm) and (70 < y < 442) and (70 < x < 442):
                rm, xm, ym = r, x, y

    # print(ym, xm, rm)

    xc, yc, r = xm, ym, rm  # location and size of the circle
    H, W = th1.shape  # size of the image
    x, y = np.meshgrid(np.arange(W), np.arange(H))  # x and y coordinates per every pixel of the image
    d2 = (x - xc) ** 2 + (y - yc) ** 2  # squared distance from the center of the circle
    mask = d2 < r ** 2  # mask is True inside of the circle

    th1[mask] = 0

    image_gray = rgb2gray(th1)
    # finding blobs
    blobs_log = blob_log(image_gray, max_sigma=10, num_sigma=10, threshold=.1)
    # radiusof blobs
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    # list containing centers of blobs
    data = []
    for blob in blobs_log:
        discard = []
        y, x, r = blob
        y = int(y)
        x = int(x)
        discard.append(x)
        discard.append(y)
        data.append(discard)
    data_r = data
    data = StandardScaler().fit_transform(data)
    # DBSCAN density based algorithm
    db = DBSCAN(eps=0.3, min_samples=5).fit(data)
    # creating a nparray with number of points set to false
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # setting true to indices of points that are core points
    core_samples_mask[db.core_sample_indices_] = True
    # labelling point to respective clusters including noise point
    labels = db.labels_

    # unique labels is number of cluster + 1(for noise is counted as a cluster)
    unique_labels = set(labels)
    # defining colors for each label

    AllRegions = []
    for k in unique_labels:
        if k == -1:
            continue
        discard = []
        b = 0
        t = 512
        r = 0
        l = 512
        for j in range(0, len(labels)):
            if labels[j] == k:
                if data_r[j][0] > r:
                    r = data_r[j][0]
                if data_r[j][0] < l:
                    l = data_r[j][0]
                if data_r[j][1] < t:
                    t = data_r[j][1]
                if data_r[j][1] > b:
                    b = data_r[j][1]
        cv2.circle(image, (l - 5, t - 5), 10, color=255, thickness=2)
        cv2.circle(image, (l - 5, b + 5), 10, color=255, thickness=2)
        cv2.circle(image, (r + 5, b + 5), 10, color=255, thickness=2)
        cv2.circle(image, (r + 5, t - 5), 10, color=255, thickness=2)
        discard.append([t - 5, l - 5])
        discard.append([b + 5, l - 5])
        discard.append([b + 5, r + 5])
        discard.append([t - 5, r + 5])
        AllRegions.append(discard)

    quality_fg = 100
    quality_bg = 20
    # print("H")
    im = Image.open(file_name)
    im = im.resize((512, 512))
    # im.save('image' +file_name[0:-4] + '_orig.jpg')
    # im.save('images/_orig.jpg')

    # print(im.size)
    mid = BytesIO()
    im.save(mid, 'JPEG', quality=quality_bg)
    ext = Image.open(mid)

    cropped_mid = BytesIO()
    im.save(cropped_mid, 'JPEG', quality=quality_fg)
    roi = Image.open(cropped_mid)

    # pixels = np.array(ext)
    ext_copy = np.array(ext)

    regions = []
    boundary = []
    for region in AllRegions:

        regions.append(Polygon(region))
        left = right = region[0][1]
        top = bottom = region[0][0]
        for i in region:
            left, right = min(left, i[1]), max(right, i[1])
            top, bottom = min(top, i[0]), max(bottom, i[0])
        boundary.append([left, top, right, bottom])

    # print(regions)
    # print(boundary)

    rois = []
    pois = []
    for i in boundary:
        r = roi.crop(i)
        rois.append(r)
        pois.append(np.array(r))

    for olc in range(len(pois)):
        for index, pixel in np.ndenumerate(pois[olc]):
            row, col, channel = index
            if channel != 0:
                continue
            point = Point(row + boundary[olc][1], col + boundary[olc][0])
            for i in regions:
                if i.contains(point):
                    ext_copy[row + boundary[olc][1], col + boundary[olc][0]] = rois[olc].getpixel((col, row))
                    break
                    print(row + boundary[olc][1], col + boundary[olc][0])
    cut_image = Image.fromarray(ext_copy)
    # cut_image.save('images/' +file_name[0:-4] + '_fin.jpg')
    temp_file_name = file_name.split('/')[-1].split('.')[0]

    cut_image.save('images/' + temp_file_name + '_compressed.jpg')
##########################################################


current_image = ""

root = tk.Tk()

T = Text(root, height=1, width=32)
T.pack()
T.insert(END, "IMAGE COMPRESSION - FUNDUS IMAGES\n\n")

photo = "home_screen.jpg"
current_image = photo 


orig_img = Image.open(photo)
resized_img = orig_img.resize((1000, 600),Image.ANTIALIAS)
root.photo = ImageTk.PhotoImage(resized_img)

vlabel=tk.Label(root,image=root.photo)
vlabel.pack()

#Browse Image Button
b2=tk.Button(root,text="Browse Image",command=change_pic)
b2.pack()


#Algo execution CLAHE
algo_button1 = tk.Button(root,text="CLAHE",command=clahe100)

algo_button1.pack(side=tk.LEFT)

#Algo execution Thresholding
algo_button2 = tk.Button(root,text="Thresholding",command=thresholding235)

algo_button2.pack(side=tk.LEFT)

#Algo execution Optic Disk detect
algo_button3 = tk.Button(root,text="Optic Disk Detection",command=detection)

algo_button3.pack(side=tk.LEFT)

#Algo execution Optic Disk removal
algo_button4 = tk.Button(root,text="Optic Disk Removal",command=removal)

algo_button4.pack(side=tk.LEFT)

#Algo execution Cluster Lessions
algo_button5 = tk.Button(root,text="Cluster Lessions",command=clusters)

algo_button5.pack(side=tk.LEFT)

#Algo execution Final Compression
algo_button5 = tk.Button(root,text="Final Compression",command=finalcompressed)

algo_button5.pack(side=tk.RIGHT)

# print("current_image is " , current_image )
root.mainloop()