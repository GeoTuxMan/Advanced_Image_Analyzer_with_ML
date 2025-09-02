import pytesseract
from PIL import Image
import cv2

# Set path to tesseract if needed:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image):
    #image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang="eng")
    print(f"OCR output: '{text}'")
    return text.strip()
