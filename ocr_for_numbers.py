import cv2
import pytesseract

# Dacă ești pe Windows, setează calea completă la executabilul Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image):
    # Conversie la grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Eliminăm zgomotul și îmbunătățim contrastul
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarizare adaptivă (funcționează mai bine pe texte izolate)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Rulează OCR specializat pe cifre
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(thresh, config=custom_config)

    print(f"OCR output: '{text}'")  # Debug în consolă
    return text.strip()
