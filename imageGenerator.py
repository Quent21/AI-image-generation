from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image as PilImage, ImageTk
from tkinter import *
from tkcolorpicker import askcolor
from time import time
from math import sqrt

W = 500
H = 500
batch_size = 128

class Image:
    def __init__(self, can, shape, width, height):
        self.can = can
        self.width = width
        self.height = height

        self.model = Sequential()
        self.model.add(InputLayer(input_shape = (2,)))
        for i in shape:
            self.model.add(Dense(i, activation = "sigmoid"))
        self.model.add(Dense(3, activation = "sigmoid"))
        self.model.summary()
        self.model.compile(loss = "mean_squared_error", optimizer = Adam(0.01))

        self.img = self.can.create_image(0, 0, anchor = "nw")

        self.constraints = []

        self.coordsNorm = np.array([[i / self.width, j / self.height] for i in range(self.width) for j in range(self.height)])
        self.coords = np.array([[i, j] for i in range(self.width) for j in range(self.height)])

    def getColors(self):
        t = time()
        res = self.model.predict(self.coordsNorm, batch_size = batch_size) * 255
        d = {(coords[0], coords[1]): color for coords, color in zip(self.coords, res)}
        arr = np.array([[d[(j, i)] for j in range(self.height)] for i in range(self.width)], dtype = np.uint8)
        self.arr = arr
        img = ImageTk.PhotoImage(image = PilImage.fromarray(arr, mode = "RGB"))
        print("Temps : " + str(round((time() - t) * 1000)) + " ms")
        return img

    def draw(self):
        self.image = self.getColors()
        self.can.itemconfig(self.img, image = self.image)

        for i in self.constraints:
            i.draw()

    def train(self, epochs = 1000):
        points = []
        for i in self.constraints:
            points.extend(i.getPoints())

        if len(points) == 0:
            return

        x = np.array([[x / self.width, y / self.height] for x, y, c in points])
        y = np.array([c.getArray() for x, y, c in points])

        self.model.fit(x, y, epochs = epochs, batch_size = batch_size)

    def showConstraints(self):
        for i in self.constraints:
            i.draw()

    def hideConstraints(self):
        for i in self.constraints:
            i.clear()

    def clearConstraints(self):
        self.hideConstraints()
        self.constraints = []

    def addPoint(self, x, y, color):
        point = Point(self.can, x, y, Color(color))
        self.constraints.append(point)
        return point

    def addLine(self, p1, p2):
        line = Line(self.can, p1, p2)
        self.constraints.append(line)
        return line

class Constraint:
    def getPoints(self):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()

class Point(Constraint):
    rad = 5

    def __init__(self, can, x, y, color):
        self.can = can
        self.x = x
        self.y = y
        self.color = color

        self.n = None

        self.draw()

    def getPoints(self):
        return [[self.x, self.y, self.color]]

    def draw(self):
        if self.n is None:
            self.n = self.can.create_oval(0, 0, 0, 0)
        self.can.coords(self.n, self.x - self.rad, self.y - self.rad, self.x + self.rad, self.y + self.rad)
        self.can.itemconfig(self.n, fill = self.color.getStr())

    def clear(self):
        self.can.delete(self.n)
        self.n = None

class Line(Constraint):
    def __init__(self, can, p1, p2):
        self.can = can
        self.p1 = p1
        self.p2 = p2

        self.n = None

        self.draw()

    def getPoints(self):
        dist = sqrt((self.p1.x - self.p2.x) ** 2 + (self.p1.y - self.p2.y) ** 2)
        steps = round(dist)
        points = []
        for i in range(steps + 1):
            coef = i / steps
            x = self.p1.x * coef + self.p2.x * (1 - coef)
            y = self.p1.y * coef + self.p2.y * (1 - coef)
            r = self.p1.color.r * coef + self.p2.color.r * (1 - coef)
            g = self.p1.color.g * coef + self.p2.color.g * (1 - coef)
            b = self.p1.color.b * coef + self.p2.color.b * (1 - coef)
            color = Color(round(r), round(g), round(b))
            points.append([round(x), round(y), color])

        return points

    def draw(self):
        if self.n is None:
            self.n = self.can.create_line(0, 0, 0, 0)
        self.can.coords(self.n, self.p1.x, self.p1.y, self.p2.x, self.p2.y)
        self.can.itemconfig(self.n, fill = self.p2.color.getStr())

    def clear(self):
        self.can.delete(self.n)
        self.n = None

class Color:
    dic = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255)
    }

    def __init__(self, r = None, g = None, b = None):
        if type(r) == np.ndarray:
            self.setArray(r)
        elif type(r) == str:
            self.setStr(r)
        elif type(r) == tuple:
            self.setRGB(*r)
        else:
            r = 0 if r == None else r
            g = 0 if g == None else g
            b = 0 if b == None else b
            self.setRGB(r, g, b)

    def getRGB(self):
        return self.r, self.g, self.b

    def setRGB(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def getArray(self):
        return np.array([self.r / 255, self.g / 255, self.b / 255])

    def setArray(self, array):
        array = array * 255
        self.setRGB(round(array[0]), round(array[1]), round(array[2]))

    def getStr(self):
        return "#%02x%02x%02x" % self.getRGB()

    def setStr(self, s):
        if s in self.dic:
            self.setRGB(*self.dic[s])
        else:
            self.setRGB(int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))

class Window:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.tk = Tk()
        self.can = Canvas(self.tk, width = self.width, height = self.height, bg = "white")
        self.can.grid(row = 0, column = 0)
        self.can.bind("<Button-1>", self.leftClick)
        self.can.bind("<Button-3>", self.rightClick)

        self.frame = Frame(self.tk)
        self.frame.grid(row = 0, column = 1)
        Button(self.frame, text = "Reset line", command = self.resetLine).grid(row = 0, column = 0)
        Button(self.frame, text = "Draw", command = self.draw).grid(row = 1, column = 0)
        Button(self.frame, text = "Learn", command = self.learn).grid(row = 2, column = 0)
        Button(self.frame, text = "Show", command = self.show).grid(row = 3, column = 0)
        Button(self.frame, text = "Hide", command = self.hide).grid(row = 4, column = 0)
        Button(self.frame, text = "Clear", command = self.clear).grid(row = 5, column = 0)

        self.img = Image(self.can, [30, 30, 30], W, H)
        self.prevPoint = None

    def leftClick(self, event):
        color = askcolor()[0]
        if color is not None:
            self.img.addPoint(event.x, event.y, color)

            self.prevPoint = None

    def rightClick(self, event):
        color = askcolor()[0]
        if color is not None:
            point = self.img.addPoint(event.x, event.y, color)

            if self.prevPoint is not None:
                self.img.addLine(self.prevPoint, point)

            self.prevPoint = point

    def resetLine(self):
        self.prevPoint = None

    def draw(self):
        self.img.draw()

    def learn(self):
        self.img.train()

    def show(self):
        self.img.showConstraints()

    def hide(self):
        self.img.hideConstraints()

    def clear(self):
        self.img.clearConstraints()
        self.prevPoint = None

    def mainloop(self):
        self.tk.mainloop()

w = Window(500, 500)
w.draw()
w.mainloop()
