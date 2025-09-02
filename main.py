import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from image_processing import load_image, to_grayscale, enhance_contrast, apply_filter
from ocr_for_numbers import extract_text_from_image
import cv2
from ml_model import IMAGE_SIZE
import joblib

class ImageAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("Advanced Image Analyzer")

        self.label = tk.Label(master, text="Upload an image to analyze")
        self.label.pack()

        self.upload_btn = tk.Button(master, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack()

        self.canvas = tk.Canvas(master, width=400, height=300)
        self.canvas.pack()

        self.filter_var = tk.StringVar(master)
        self.filter_var.set("none")
        self.filter_menu = tk.OptionMenu(master, self.filter_var, "none", "sobel", "gaussian", "sharpen")
        self.filter_menu.pack()

        self.apply_btn = tk.Button(master, text="Apply Filter", command=self.apply_filter)
        self.apply_btn.pack()

        self.ocr_btn = tk.Button(master, text="Extract Text", command=self.extract_text_from_image)
        self.ocr_btn.pack()

        btn_classify = tk.Button(master, text="Classify Image", command=self.classify_image)
        btn_classify.pack(pady=5)
        
        self.image = None
        self.tk_image = None
        
        self.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.bmp")])
        if file_path:
            self.image = load_image(file_path)
            self.display_image(self.image)

    def display_image(self, image):
        pil_image = Image.fromarray((image*255).astype('uint8')) if image.max() <= 1 else Image.fromarray(image)
        pil_image = pil_image.resize((400,300))
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def apply_filter(self):
        if self.image is not None:
            filtered = apply_filter(to_grayscale(self.image), self.filter_var.get())
            self.display_image(filtered)

    def extract_text_from_image(self):
        if self.image is not None:
            cv_img = (self.image*255).astype('uint8') if self.image.max() <= 1 else self.image
            text = extract_text_from_image(cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Extracted Text", text)
            
    def classify_image(self):
        if self.image is not None:
            # incarca modelul
            clf, classes = joblib.load("image_classifier.pkl")
        
            # Foloseste imaginea deja incarcata
            img = self.image
            # Daca e float [0,1], transforma in uint8
            if img.max() <= 1:
                img = (img * 255).astype('uint8')

            # Resize la dimensiunea asteptata de model
            img = cv2.resize(img, IMAGE_SIZE)
            img_flat = img.flatten().reshape(1, -1)
        
            # predictie
            pred = clf.predict(img_flat)[0]
            predicted_class = classes[pred]
        
            messagebox.showinfo("Classification Result", f"This looks like a: {predicted_class}")
        else:
            messagebox.showwarning("Warning", "Please upload an image first!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnalyzerApp(root)
    root.mainloop()
