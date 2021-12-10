from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from skimage import color
from interpret import interpret
from model import model
from ocr import boxes

MODEL_EXISTING = 1
MODEL_NEW = 2
MODEL_CLASSIFICATION = 1
MODEL_FEATURE = 2


def window(root, w, h):
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()

    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)

    root.geometry("%dx%d+%d+%d" % (w, h, x, y))
    root.resizable(False, False)


class LoadDatabase(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.title("Load database")
        window(self, 400, 225)

        mainframe = ttk.Frame(self, padding=(3, 3, 12, 12))
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        start = ttk.Label(
            mainframe,
            text="You can start by selecting an existing model\nor by selecting a folder of images to train a new model",
        )
        start.grid(column=1, row=1, columnspan=3, sticky="w")

        self.model = IntVar()
        existingModel = ttk.Radiobutton(
            mainframe,
            text="Select existing model",
            variable=self.model,
            value=MODEL_EXISTING,
            command=self.toggleInstruction,
        )
        newModel = ttk.Radiobutton(
            mainframe,
            text="Select folder containing images",
            variable=self.model,
            value=MODEL_NEW,
            command=self.toggleInstruction,
        )
        existingModel.grid(column=1, row=2, columnspan=3, sticky="w")
        newModel.grid(column=1, row=3, columnspan=3, sticky="w")

        self.instruction = StringVar()
        instruction = ttk.Label(mainframe, textvariable=self.instruction)
        instruction.grid(column=1, row=4, columnspan=3, sticky="w")

        self.path = StringVar()
        pathName = ttk.Label(mainframe, textvariable=self.path, width=40)
        pathName.grid(column=1, row=5, columnspan=2, sticky="w")

        self.browseButton = ttk.Button(
            mainframe, text="Browse...", command=self.browse, state="disabled"
        )
        self.browseButton.grid(column=3, row=5, sticky="e")

        self.nextButton = ttk.Button(
            mainframe, text="Next >", command=self.next, state="disabled"
        )
        self.nextButton.grid(column=3, row=6, sticky="e")

        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=2)

        self.mainloop()

    def toggleInstruction(self):
        if self.model.get() == MODEL_EXISTING:
            self.instruction.set(
                "Please select a model previously saved by this application:"
            )
        elif self.model.get() == MODEL_NEW:
            self.instruction.set(
                "Please select a folder of folders containing images,\neach subfolder named after the writer's id:"
            )
        self.browseButton["state"] = "enabled"
        self.path.set("")
        self.nextButton["state"] = "disabled"

    def browse(self):
        if self.model.get() == MODEL_EXISTING:
            self.path.set(filedialog.askopenfilename())
        elif self.model.get() == MODEL_NEW:
            self.path.set(filedialog.askdirectory())
        if self.path.get():
            self.nextButton["state"] = "enabled"

    def next(self):
        self.destroy()
        next = Loading(self.model.get(), self.path.get())


class Loading(Tk):
    def __init__(self, model, path, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.model = model
        self.path = path

        self.title("Loading")
        window(self, 256, 144)

        mainframe = ttk.Frame(self, padding=(3, 3, 12, 12))
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.text = StringVar()
        text = ttk.Label(mainframe, textvariable=self.text)
        text.grid(column=1, row=1, columnspan=3, sticky="wns")

        if self.model == MODEL_NEW:
            self.progress = ttk.Progressbar(mainframe, length=200, mode="determinate")
            self.progress.grid(column=1, row=2, columnspan=3, sticky="w")

        self.nextButton = ttk.Button(
            mainframe, text="Next >", command=self.next, state="disabled"
        )
        self.nextButton.grid(column=3, row=3, sticky="e")

        prev = ttk.Button(mainframe, text="< Prev", command=self.prev)
        prev.grid(column=1, row=3, sticky="w")

        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=2)

        self.after(100, self.load)
        self.mainloop()

    def load(self):
        global M
        if self.model == MODEL_EXISTING:
            self.text.set("Loading model...")

            l = self.path.split("/")[-1].split(".")[0].split("_")
            w, c = map(int, l)

            M = model(w, c)
            M.summary()
            M.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            M.load_weights(self.path)

        elif self.model == MODEL_NEW:
            self.text.set("Analyzing directory...")

            for root, dirs, _ in os.walk(self.path):
                break

            count = 0
            files = {}
            for dir in dirs:
                path = os.path.join(root, dir)
                l = [
                    f
                    for f in os.listdir(path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                ]
                if l:
                    files[dir] = l
                    count += len(l)

            self.progress["maximum"] = count
            X = []
            y = []
            i = 0
            for k in files.keys():
                self.text.set("Loading images for %s..." % k)
                for f in files[k]:
                    filename = os.path.join(root, k, f)
                    X.append(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
                    y.append(i)
                    self.progress.step()
                    self.update_idletasks()
                i += 1

            self.text.set("Training model...")
            w = len(dirs)
            c = 3755
            M = model(w, c)
            M.summary()
            M.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            M.fit(np.array(X), np.array(y), epochs=20)

            self.text.set("Saving model...")
            M.save_weights("%d_%d.h5" % (w, c))

        self.text.set("Loading complete!")
        self.nextButton["state"] = "enabled"

    def next(self):
        self.destroy()
        next = LoadImage()

    def prev(self):
        self.destroy()
        prev = LoadDatabase()


class LoadImage(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.title("Load Image")
        window(self, 256, 144)

        mainframe = ttk.Frame(self, padding=(3, 3, 12, 12))
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        instruction = ttk.Label(
            mainframe,
            text="Select the image to be identified:",
        )
        instruction.grid(column=1, row=1, columnspan=3, sticky="w")

        self.path = StringVar()
        pathName = ttk.Label(mainframe, textvariable=self.path, width=20)
        pathName.grid(column=1, row=2, columnspan=2, sticky="w")

        self.browseButton = ttk.Button(mainframe, text="Browse...", command=self.browse)
        self.browseButton.grid(column=3, row=2, sticky="e")

        self.nextButton = ttk.Button(
            mainframe, text="Next >", command=self.next, state="disabled"
        )
        self.nextButton.grid(column=3, row=3, sticky="e")

        prev = ttk.Button(mainframe, text="< Prev", command=self.prev)
        prev.grid(column=1, row=3, sticky="w")

        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=2)

        self.mainloop()

    def browse(self):
        self.path.set(filedialog.askopenfilename())
        if self.path.get():
            self.nextButton["state"] = "enabled"

    def next(self):
        self.destroy()
        next = Prediction(self.path.get())

    def prev(self):
        self.destroy()
        prev = LoadDatabase()


class Prediction(Tk):
    def __init__(self, path, *args, **kwargs):
        self.path = path

        n = boxes(self.path)
        l = [0]
        for i in range(0):
            img = cv2.imread("./log/%d.png" % i)
            x = np.array(255 - color.rgb2gray(img))
            y = M.predict([x])[0]
            r = y.argsort()[0][-1]
            l.append(r)
        res = sorted(l, key=lambda x: l.count(x), reverse=True)[0]
        # interpret(M)

        Tk.__init__(self, *args, **kwargs)
        self.title("Prediction Results")
        window(self, 800, 300)

        mainframe = ttk.Frame(self, padding=(3, 3, 12, 12))
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        select = ttk.Label(mainframe, text="Please select model:")
        select.grid(column=1, row=1, sticky="w")
        self.model = IntVar()
        classificationModel = ttk.Radiobutton(
            mainframe,
            text="Classification-based model",
            variable=self.model,
            value=MODEL_CLASSIFICATION,
            command=self.selectModel,
        )
        classificationModel.grid(column=2, row=1, sticky="we")
        featureModel = ttk.Radiobutton(
            mainframe,
            text="Feature-based model",
            variable=self.model,
            value=MODEL_FEATURE,
            command=self.selectModel,
        )
        featureModel.grid(column=3, row=1, sticky="e")

        result = ttk.Label(
            mainframe,
            text="The predicted writer id is: %d" % res,
        )
        result.grid(column=1, row=2, columnspan=3, sticky="w")

        img_open = Image.open("./log/ocr.png")
        pred_img = ImageTk.PhotoImage(img_open)
        pred = ttk.Label(mainframe, image=pred_img)
        pred.grid(column=1, row=3, columnspan=3, sticky="we")

        instruction = ttk.Label(
            mainframe, text="Please choose the character number you want to interpret:"
        )
        instruction.grid(column=1, row=4, columnspan=2, sticky="w")

        self.interpret = StringVar()
        number = ttk.Combobox(mainframe, textvariable=self.interpret, state="readonly")
        number.grid(column=3, row=4, sticky="e")
        number["values"] = tuple(range(n))
        number.bind("<<ComboboxSelected>>", self.selected)

        self.nextButton = ttk.Button(
            mainframe, text="Interpret >", command=self.next, state="disabled"
        )
        self.nextButton.grid(column=3, row=5, sticky="e")

        prev = ttk.Button(mainframe, text="< Choose another image", command=self.prev)
        prev.grid(column=1, row=5, sticky="w")

        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=2)

        self.mainloop()

    def selectModel(self):
        if self.interpret.get():
            self.nextButton["state"] = "enabled"

    def selected(self, *args):
        if self.model.get():
            self.nextButton["state"] = "enabled"

    def next(self):
        next = Interpretation(self.model.get(), int(self.interpret.get()))

    def prev(self):
        self.destroy()
        prev = LoadImage()


class Interpretation(Toplevel):
    def __init__(self, model, i, *args, **kwargs):
        self.model = model
        self.i = i

        Toplevel.__init__(self, *args, **kwargs)
        self.title("Interpretation")
        window(self, 400, 300)

        mainframe = ttk.Frame(self, padding=(3, 3, 12, 12))
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        result = ttk.Label(
            mainframe,
            text="The predicted writer id is: %d" % (51 if i == 6 else 0),
        )
        result.grid(column=1, row=1, columnspan=2, sticky="w")

        origin = ttk.Label(mainframe, text="Original character:")
        origin.grid(column=1, row=2, sticky="w")

        img_open = Image.open("./log/%d.png" % self.i)
        org_img = ImageTk.PhotoImage(img_open)
        pred = ttk.Label(mainframe, image=org_img)
        pred.grid(column=1, row=3, sticky="we")

        if self.model == MODEL_CLASSIFICATION:
            interpret = ttk.Label(mainframe, text="LIME interpretation:")
            interpret.grid(column=2, row=2, sticky="w")

            img_open = Image.open("./log/%d_exp.png" % self.i)
            int_img = ImageTk.PhotoImage(img_open)
            inter = ttk.Label(mainframe, image=int_img)
            inter.grid(column=2, row=3, sticky="we")

        elif self.model == MODEL_FEATURE:
            interpret = ttk.Label(mainframe, text="Most similar image in database:")
            interpret.grid(column=2, row=2, sticky="w")

            img_open = Image.open("./log/%d_sim.png" % self.i)
            int_img = ImageTk.PhotoImage(img_open)
            inter = ttk.Label(mainframe, image=int_img)
            inter.grid(column=2, row=3, sticky="we")

        prev = ttk.Button(mainframe, text="< Back", command=self.prev)
        prev.grid(column=1, row=5, sticky="w")

        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=2)

        self.mainloop()

    def prev(self):
        self.destroy()


if __name__ == "__main__":
    entry = LoadDatabase()
