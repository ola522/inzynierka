#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import matplotlib .pyplot as plt
import matplotlib.image as mpimg
from skimage import io
from skimage.color import rgb2gray
import scipy.signal as signal
import scipy.ndimage as nd
from numpy import asarray
from skimage import feature
import os
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
import math


# In[17]:


path = r'C:\Users\olate\Desktop\abc\4\odcisk.jpg' #ścieżka do obrazu
im = Image.open(path)
plt.imshow(im, cmap = "gray")


# # Obrót obrazu

# In[3]:


a = 10
#a = 9.01525861876796
im = im.rotate(a)          #obrót obrazu
#plt.figure(dpi=250)
plt.imshow(im, cmap="gray")
#plt.savefig("obrocony_obraz.jpg")


# # Wyznaczenie krawędzi

# In[18]:


image = im
image = np.array(im)
for i in range(np.shape(image)[0]):
    for j in range(np.shape(image)[1]):
        image[i, j] = 255 - image[i, j]     #odwrócenie kolorów obrazu

plt.imshow(image, cmap = "gray")


# In[19]:


can = feature.canny(image, sigma = 3)  #wyznaczene krawędzi
can = can.astype(int)
    
for i in range(1, np.size(can, 0)):
    for j in range (1, np.size(can, 1)):
        if (can[i,j]==0):
            can[i,j]=1
        else:
            can[i,j] = 0

can = can[5:-5, 5:-5]
plt.axis("off")
plt.imshow(can, cmap = "gray")


# # Wyznaczenie punktów charakterystycznych

# In[20]:


pieta = can[301:, :]
wysklepienie = can[201:300, :]
przodostopie = can[136:200, :]
palce = can[:136, :]
palce2=can[:131, 130:]

plt.figure(dpi=250)
plt.subplot(1,4,1)
plt.title("Palce")
plt.axis("off")
plt.imshow(palce, cmap="gray")
plt.subplot(1,4,2)
plt.title("Przodostopie")
plt.axis("off")
plt.imshow(przodostopie, cmap="gray")
plt.subplot(1,4,3)
plt.title("Wysklepienie")
plt.axis("off")
plt.imshow(wysklepienie, cmap="gray")
plt.subplot(1,4,4)
plt.title("Pięta")
plt.axis("off")
plt.imshow(pieta, cmap="gray")
#plt.savefig("czesci.jpg")


# In[21]:


def point(image, mode):    #funkcja wyznaczająca punkty charakterystyczne
    b=0
    i = j = 0
    x = image.shape[0]
    y = image.shape[1]
    if (mode=="up"):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if (image[i,j]==0):
                    return i, j
    if (mode=="left"):
        for j in range(image.shape[1]):
            for i in range(image.shape[0]):
                if (image[i,j]==0):
                    return i, j
    if (mode=="down"):
        for i in range(x-1, -1, -1):
            for j in range(y-1, -1, -1):
                if (image[i,j]==0):
                    return i, j
    if (mode=="right"):
        for j in range(y-1, -1, -1):
            for i in range(x-1, -1, -1):
                if (image[i,j]==0):
                    return i, j
    return i, j


# In[22]:


Ay, Ax = point(can, "up")
A = (Ax, Ay)
By, Bx = point(can, "down")
Cy, Cx = point(przodostopie, "left")
Cy = Cy + 131
Dy, Dx = point(przodostopie, "right")
Dy = Dy + 131
Sy, Sx = point(pieta, "left")
Sy = Sy +301
Ty, Tx = point(pieta, "right")
Ty = Ty + 301
Tx = Tx
Ey, Ex = point(palce, "left")
#Sy = Sy + 201
Fy, Fx = point(palce, "right")
Fy = Fy
YY, XX = point(palce2, "up")
XX = XX + 130

plt.figure(dpi=200)
plt.plot(A[0], A[1], "ro")
plt.annotate("A", (A[0], A[1]))
plt.plot(Bx, By, "ro")
plt.annotate("B", (Bx, By))
plt.plot(Cx, Cy, "ro")
plt.annotate("C", (Cx, Cy))
plt.plot(Dx, Dy, "ro")
plt.annotate("D", (Dx, Dy))
plt.plot(Sx, Sy, "ro")
plt.annotate("S", (Sx, Sy))
plt.plot(Tx, Ty, "ro")
plt.annotate("T", (Tx, Ty))
plt.plot(Ex, Ey, "ro")
plt.annotate("E", (Ex, Ey))
plt.plot(Fx, Fy, "ro")
plt.annotate("F", (Fx, Fy))
plt.plot(XX, YY, "ro")
plt.annotate("P", (XX, YY))
plt.imshow(can, cmap="gray")
#plt.savefig("Punkty.jpg")
plt.show()


# In[23]:


plt.figure(dpi=200)

plt.plot((A[0], Bx), (A[1], By), 'bo', linestyle="--")
plt.plot((Cx, Dx), (Cy, Dy), 'go', linestyle="--")
plt.plot((Sx, Tx), (Sy, Ty), 'yo', linestyle="--")
plt.annotate("A'", (A[0], A[1]))
plt.annotate("B'", (Bx, By))
plt.annotate("C'", (Cx, Cy))
plt.annotate("D'", (Dx, Dy))
plt.annotate("S'", (Sx, Sy))
plt.annotate("T'", (Tx, Ty))
plt.plot(Ex, Ey, "ro")
plt.annotate("E'", (Ex, Ey))
plt.plot(Fx, Fy, "ro")
plt.annotate("F'", (Fx, Fy))
plt.plot(XX, YY, "ro")
plt.annotate("P'", (XX, YY))
plt.plot((Cx, Sx), (Cy, Sy), 'yo', linestyle="--")
plt.plot((Tx, Dx), (Ty, Dy), 'ro', linestyle="--")
plt.plot((Cx, Ex), (Cy, Ey), 'mo', linestyle="--")
plt.plot((Fx, Dx), (Fy, Dy), 'mo', linestyle="--")
plt.imshow(can, cmap="gray")
#plt.savefig("Linie_prim.jpg")


# # Wyznaczone wskaźniki

# In[24]:


AB = np.sqrt((A[0]+Bx)^2+(A[1]+By)^2)
CD = np.sqrt((Cx+Dx)^2+(Cy+Dy)^2)
Wejsflog = AB/CD

if (Wejsflog > 2.3):
    print("Wskaźnik Wejsfloga: ", round(Wejsflog, 2), ". Stopa jest wysklepiona prawidłowo.")
else:
    print("Wskaźnik Wejsfloga: ", round(Wejsflog, 2), ". Stopę charakteryzuje wysklepienie poprzeczne.")


# In[25]:


def lineFromPoints(P, Q):
    a = (Q[1] - P[1])/(Q[0] - P[0])
    b = P[1] - P[0]*a
    return a

def angleBetween2Lines(a1, a2):
    tan = np.absolute((a1 - a2)/(1 + a1*a2))
    theta = np.arctan(tan)
    theta = np.rad2deg(theta)
    return theta


# In[26]:


aCS = lineFromPoints((Cx, Cy), (Sx, Sy))
aDT = lineFromPoints((Dx, Dy), (Tx, Ty))
theta_CS_DT = angleBetween2Lines(aCS, aDT)

if (theta_CS_DT >= 15 and theta_CS_DT <= 18):
    print("Kąt Gamma wynosi: ", round(theta_CS_DT, 2), " Kąt jest prawidłowy")
elif (theta_CS_DT < 15):
    print("Kąt Gamma wynosi: ", round(theta_CS_DT, 2), " Kąt jest za mały")
elif (theta_CS_DT > 18):
    print("Kąt Gamma wynosi: ", round(theta_CS_DT, 2), " Kąt jest za duży")


# In[27]:


aCS = lineFromPoints((Cx, Cy), (Sx, Sy))
aCE = lineFromPoints((Cx, Cy), (Ex, Ey))
theta_CS_CE = angleBetween2Lines(aCS, aCE)

if (theta_CS_CE >= 0 and theta_CS_CE <= 9):
    print("Kąt Alfa wynosi: ", round(theta_CS_CE, 2), " Kąt jest prawidłowy")
elif (theta_CS_CE > 9):
    print("Kąt Alfa wynosi: ", round(theta_CS_CE, 2), " Kąt jest za duży")


# In[28]:


aDT = lineFromPoints((Dx, Dy), (Tx, Ty))
aDF = lineFromPoints((Dx, Dy), (Fx, Fy))
theta_DF_DT = angleBetween2Lines(aDF, aDT)
if (theta_DF_DT >= 0 and theta_DF_DT <= 9):
    print("Kąt Beta wynosi: ", round(theta_DF_DT, 2), " Kąt jest prawidłowy")
elif (theta_DF_DT > 9):
    print("Kąt Beta wynosi: ", round(theta_DF_DT, 2), " Kąt jest za duży")


# # Kąt rotacji stopy

# In[36]:


aXB = lineFromPoints((XX, YY), (Bx, By))
rotation = math.atan(aXB)           # Obliczenie kąta odchylenia w radianach
rotation = math.degrees(rotation)
rotation = 180 - (90 - rotation)
print(rotation)


# In[42]:


plt.figure(dpi=250)
plt.imshow(im, cmap="gray")
plt.axis("off")
label1 = "Kąt rotacji = " + str(round(rotation, 2)) + "\u00b0"
plt.plot((XX, Bx), (YY, By), "g-", label = label1)
plt.axvline(x = Bx, label = "Linia pionowa")
plt.legend(loc = "lower right", fontsize = "small")
#plt.savefig("kat1.jpg")


# # Momenty bezwładności - potrzebne funkcje

# In[87]:


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def normal(image):
    image = rgb2gray(image)
    opened_image = image
    #kernel = np.ones((15, 15), np.uint8)
    #opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    #kernel = np.ones((5, 5), np.uint8)
    #opened_image = cv2.morphologyEx(opened_image, cv2.MORPH_OPEN, kernel, iterations=3)
    opened_image = gaussian_filter(opened_image, sigma=1) 
    
    min_val = np.min(opened_image)
    max_val = np.max(opened_image)
    for i in range (0, np.shape(opened_image)[0]):
        for j in range(0, np.shape(opened_image)[1]):
            opened_image[i, j] = (opened_image[i, j] - min_val)/(max_val - min_val)

    return opened_image

def binary(image):
    xyz, binary_image = cv2.threshold(image, .6, 1, cv2.THRESH_BINARY)
    binary_image = binary_image.astype(int)
    for i in range(0, 20):
        for j in range(0, 20):
            binary_image[i, j] = 0
    
    return binary_image

def total_footprint(images):  
    if str(type(images[0])) == "<class 'PIL.JpegImagePlugin.JpegImageFile'>":
        image = np.array(images[0])
        l = len(images)
        for i in range(0, l):
            images2[i] = rgb2gray(images[i])
            arr_im = np.array(images2[i])
            image = image + arr_im
        min_val = np.min(image)
        max_val = np.max(image)
    else:
        image = images[0]
        for i in range (0, len(images)):
            image = image + images[i]
            
    return image

def srodek_ciezkosci(image):
    sumx = 0
    sumy = 0
    pol = 0
    size0 = np.shape(image)[0]
    size1 = np.shape(image)[1]
    for i in range(0, size0):
        for j in range(0, size1):
            if (image[i,j] > 0):
                sumx += j
                sumy += i
                pol = pol+1
    if (pol!=0):
        sumx = sumx/pol
        sumy = sumy/pol
    return sumx, sumy

def moment_b(image):
    Ix = 0
    Iy = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i,j] == 0):
                Ix += j*j
                Iy += i*i
    return Ix, Iy

def moment_b_osie_centralne(image):
    Ix0 = 0
    Iy0 = 0
    xend, a = point(image, "down")
    b, yend = point(image, "right")
    srodek_x, srodek_y = srodek_ciezkosci(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i,j] == 1):
                Ix0 += (srodek_x - i - xend)**2
                Iy0 += (srodek_y - j - yend)**2
    return Ix0, Iy0

def moment_dew(image):
    Ixy = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i,j] == 0):
                Ixy += i*j
    return Ixy

def moment_dew_centralne(image):
    Ixy = 0
    srodek_x, srodek_y = srodek_ciezkosci(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i,j] == 1):
                Ixy += np.abs((srodek_x - i)*(srodek_y - j))
    return Ixy

def g_moment(ix, iy, ixy):
    i1 = ((ix+iy)/2)+np.sqrt((((ix-iy)/2)**2)+ixy**2)
    i2 = ((ix+iy)/2)-np.sqrt((((ix-iy)/2)**2)+ixy**2)
    phi_rad = np.arctan(2*ixy/(ix-iy))/2
    phi = phi_rad*180/np.pi
    return i1, i2, phi, phi_rad

def g_moment2(ix, iy, ixy):
    phi_rad = np.arctan(2*ixy/(ix-iy))/2
    phi = phi_rad*180/np.pi
    i1 = ((ix+iy)/2)+((ix-iy)/2)*np.cos(2*phi)+ixy*np.sin(2*phi)
    i2 = ((ix+iy)/2)-((ix-iy)/2)*np.cos(2*phi)-ixy*np.sin(2*phi)

    return i1, i2, phi, phi_rad


def test_sh(shape, mode):
    x1, y1 = srodek_ciezkosci(shape)
    ix0, iy0 = moment_b_osie_centralne(shape)
    ixy0 = moment_dew_centralne(shape)
    i1, i2, phi, phi_rad = g_moment(ix0, iy0, ixy0)
    axx = np.linspace(0, 1200, 50)
    axy_mom = np.tan(phi_rad) * (axx - x1) + y1
    a = np.radians(90)
    axy_mom2 = -np.tan(a - phi_rad) * (axx - x1) + y1
    if (mode == "y"):
        plt.figure(dpi = 400)
        plt.plot(axx, axy_mom)
        plt.plot(axx, axy_mom2)
        plt.plot(x1, y1, "ro")
        plt.axis("off")
        plt.imshow(shape, cmap="gray")
        plt.savefig("elipsa2.jpg")
    else:
        print(i1, i2, ix0, iy0, ixy0, phi)
        plt.plot(axx, axy_mom)
        plt.plot(axx, axy_mom2)
        plt.plot(x1, y1, "ro")
        plt.imshow(shape, cmap="gray")

def oblicz_momenty(obraz_binarny):
    momenty = {'m00': 0, 'm10': 0, 'm01': 0, 'm20': 0, 'm11': 0, 'm02': 0}

    for y in range(obraz_binarny.shape[0]):
        for x in range(obraz_binarny.shape[1]):
            if obraz_binarny[y, x] > 0:  # Biały piksel
                momenty['m00'] += 1
                momenty['m10'] += x
                momenty['m01'] += y
                momenty['m20'] += x * x
                momenty['m11'] += x * y
                momenty['m02'] += y * y

    return momenty

def znajdz_osie_centralne_odcisku(obraz_binarny):
    momenty = oblicz_momenty(obraz_binarny)
    
    # Centrum masy
    x_c = momenty['m10'] / momenty['m00']
    y_c = momenty['m01'] / momenty['m00']
    
    # Momenty centralne
    u20 = momenty['m20'] / momenty['m00'] - x_c * x_c
    u02 = momenty['m02'] / momenty['m00'] - y_c * y_c
    u11 = momenty['m11'] / momenty['m00'] - x_c * y_c
    
    # Współczynniki
    a = np.sqrt(2 * (u20 + u02 + np.sqrt(u20 ** 2 + u02 ** 2 - 2 * u20 * u02 + 4 * u11 ** 2)))
    b = np.sqrt(2 * (u20 + u02 - np.sqrt(u20 ** 2 + u02 ** 2 - 2 * u20 * u02 + 4 * u11 ** 2)))
    
    # Kąt obrotu 
    if u20 - u02 != 0:
        theta = 0.5 * np.arctan((2 * u11) / (u20 - u02))
    else:
        theta = np.pi / 4
    
    print(momenty['m20'], momenty['m02'], momenty['m11'])
    
    return a, b, theta

def test2(image):
    dluzsza_os, krotsza_os, kat_obrotu = znajdz_osie_centralne_odcisku(image)
    kat = np.degrees(kat_obrotu)
    print(kat)
    x1, y1 = srodek_ciezkosci(image)
    axx = np.linspace(0, 1200, 50)
    axy_mom = np.tan(kat_obrotu) * (axx - x1) + y1
    a = np.radians(90)
    axy_mom2 = -np.tan(a - kat_obrotu) * (axx - x1) + y1
    #plt.figure(dpi = 400)
    plt.plot(axx, axy_mom)
    plt.plot(axx, axy_mom2)
    plt.plot(x1, y1, "ro")
    plt.axis("off")
    plt.imshow(image, cmap="gray")
    #plt.savefig("momenty3.jpg")


# # Wstępne przetwarzanie

# In[89]:


path = r'C:\Users\olate\Desktop\abc\wszystkie'
images = load_images_from_folder(path)
script_dir = os.path.dirname(r'C:\Users\olate\Desktop')
rel_path = r'C:\Users\olate\Desktop\abc\wszystkie'
abs_file_path = os.path.join(script_dir, rel_path)

#SORTOWANIE
num = 0
num_of_frames = 150
sorted_images = images

for i in range(0, num_of_frames+1):
    current_file ="\\4frame" + str(i) +".jpg"
    path = abs_file_path+current_file

    if os.path.exists(path):
        image = Image.open(abs_file_path+current_file, 'r')
        sorted_images[num] = image
        num += 1


opened_im = sorted_images
canny_im = []
for i in range(0, len(sorted_images)):
    opened_im[i] = normal(sorted_images[i])


binary_im = opened_im
for i in range(0, len(opened_im)):
    binary_im[i] = binary(opened_im[i])
    
ima = total_footprint(images)


# In[90]:


ima = im
ima = np.array(ima)

for i in range(np.shape(ima)[0]):
    for j in range(np.shape(ima)[1]):
        if (ima[i,j] > 0):
            ima[i,j] == 1
        else:
            ima[i, j] == 0


# In[91]:


srodekx, srodeky = srodek_ciezkosci(ima)
print(srodekx)
plt.imshow(ima, cmap = "gray")
plt.scatter(srodekx, srodeky)


# # Środki ciężkości

# In[93]:


srodek_x = []
srodek_y = []
for i in range(0, len(opened_im)):
    result1, result2 = srodek_ciezkosci(opened_im[i])
    srodek_x.append(result1)
    srodek_y.append(result2)

total_im = total_footprint(binary_im)
total_x, total_y = srodek_ciezkosci(total_im)


# In[94]:


x =[]
y = []
for i in range(0, len(srodek_x)):
    if srodek_x[i]!=0 or srodek_y[i]!=0:
        x.append(srodek_x[i])
        y.append(srodek_y[i])


# In[96]:


#plt.figure(dpi=250)
plt.scatter(x, y, s=5)
plt.imshow(total_im, cmap="gray")
#plt.savefig("srodki_ciezkosci.jpg")


# # Testy na figurach i odcisku

# In[73]:


rect = np.zeros((200, 200))
for i in range(rect.shape[0]):
    for j in range(rect.shape[1]):
        if i >=50 and i<=150 and j>=70 and j<=100:
            rect[i, j] = 1

p = r'C:\Users\olate\Desktop\inzynierka\rect.jpg'
rect = Image.open(p)
rect = rect.rotate(5)
rect = asarray(rect)
r = normal(rect)
r = binary(r)
test2(r)
#plt.imshow(rect, cmap ="gray")


# In[74]:


triangle = np.zeros((200, 200))
x1, y1 = 50, 150
x2, y2 = 50, 50
x3, y3 = 150, 50

for j in range(triangle.shape[0]):
    for i in range(triangle.shape[1]):
        if (
            (y2 - y1) * (i - x1) - (x2 - x1) * (j - y1) > 0 and
            (y3 - y2) * (i - x2) - (x3 - x2) * (j - y2) > 0 and
            (y1 - y3) * (i - x3) - (x1 - x3) * (j - y3) > 0
        ) or (
            (y2 - y1) * (i - x1) - (x2 - x1) * (j - y1) < 0 and
            (y3 - y2) * (i - x2) - (x3 - x2) * (j - y2) < 0 and
            (y1 - y3) * (i - x3) - (x1 - x3) * (j - y3) < 0
        ):
            triangle[i, j] = 1

#plt.imshow(triangle, cmap="gray")
p = r'C:\Users\olate\Desktop\inzynierka\t.jpg'
t = Image.open(p)
t = t.rotate(5)
t = asarray(t)
t = normal(t)
t = binary(t)
test2(t)
#plt.imshow(t, cmap ="gray")


# In[76]:


test2(ima)


# # Pole powierzchni

# In[77]:


def surface_area(image):
    total_area = np.shape(image)[0]*np.shape(image)[1]
    area = 0
    new = []
    max_val = np.max(image)
    for i in range(np.shape(image)[0]):
        for j in range (np.shape(image)[1]):
            if (image[i,j] > .75*max_val):
                area +=1
   
    if (area!=0):
        result = area*100/total_area
        result = round(result, 2)
    else:
        result = 0
    return result


# In[78]:


import timeit
area = opened_im
start = timeit.timeit()
for i in range(0, len(opened_im)):
    area[i] = surface_area(opened_im[i])

end = timeit.timeit()
print(start)
print(end)


# In[80]:


plt.plot(area)
plt.xlabel("Czas [s/120]")
plt.ylabel("Powierzchnia odcisku")
#plt.savefig("wyk_pow.jpg")


# In[ ]:




