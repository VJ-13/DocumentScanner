import cv2 as cv
import numpy as np
from imutils.perspective import four_point_transform
import pytesseract

# Initializing Variables
WIDTH, HEIGHT = 2480, 3050
count = 0
scale = 0.4
font = cv.FONT_HERSHEY_SIMPLEX
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Webcam
webcam = False
capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Image Path
image_path = 'Images/img7.jpg'
 
def scan_detection(image):
    # To allow other functions to access the variable
    global document_contour
    
    # Image Boundary Points
    document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])

    # Changing the Image to Grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Adding Gaussian Blur
    blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)

    # Adding the Threshold
    _, threshold = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Finding the Contours
    contours, _ = cv.findContours(threshold, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Sorting the Contours by Area in Descending Order
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    # Finding the Largest Contour with 4 Corners
    max_area = 0
    # Goes through every contour and checks which one is the biggest
    for contour in contours:
        area = cv.contourArea(contour)
        # Gets the area that is bigger than 1000
        if area > 1000:
            # Finds the perimeter of the contour and makes it closed
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.015 * peri, True)
            # Checks if the approximated contour has 4 points and has the max area
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area

    # Drawing the Contour
    cv.drawContours(img, [document_contour], -1, (0, 255, 0), 3)


def center_text(image, text):
    text_size = cv.getTextSize(text, font, 2, 5)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    cv.putText(image, text, (text_x, text_y), font, 2, (0, 255, 0), 5, cv.LINE_AA)

# scan6.jpg, scan 7.jpg 
while True:

    # Capture the Video or Image
    if webcam:
        isTrue, img = capture.read()
    else:
        img = cv.imread(image_path)

    # Resize
    img = cv.resize(img, (WIDTH, HEIGHT), interpolation=cv.INTER_CUBIC)
    img_copy = img.copy()

    # Scan Detection
    scan_detection(img_copy)

    cv.imshow("Original Image", cv.resize(img, (int(scale * WIDTH), int(scale * HEIGHT))))

    # Warping the Image by using the 4 points of the document contour
    imgWarped = four_point_transform(img_copy, document_contour.reshape(4, 2))
    cv.imshow("Warped Image", cv.resize(imgWarped, (int(scale * imgWarped.shape[1]), int(scale * imgWarped.shape[0]))))

    # Image Processing for better results
    gray = cv.cvtColor(imgWarped, cv.COLOR_BGR2GRAY)
    _, threshold = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)
    
    # Cropping the Image to make sure the edges are not included
    finalImage = threshold[10:threshold.shape[0] - 10, 10:threshold.shape[1] - 10]

    # Displaying the Final Image
    cv.imshow("Final Image", cv.resize(finalImage, (int(scale * finalImage.shape[1]), int(scale * finalImage.shape[0]))))

    pressed_key = cv.waitKey(1) & 0xFF

    # Press ESC to exit the program
    if pressed_key == 27:
        break

    # Press 's' to save the scanned image
    elif pressed_key == ord('s'):
        cv.imwrite("Scanned/scanned_" + image_path.split('/')[1].split('.')[0] + " - " + str(count) + ".jpg", finalImage)
        count += 1
        center_text(img, "Scan Saved")
        cv.imshow("Original Image", cv.resize(img, (int(scale * WIDTH), int(scale * HEIGHT))))
        cv.waitKey(500)

    elif pressed_key == ord('o'):
        file = open("Output/ocr_" + image_path.split('/')[1].split('.')[0] + " - " + str(count - 1) + ".txt", "w")
        ocr = pytesseract.image_to_string(imgWarped)
        # print(ocr)
        file.write(ocr)
        file.close()
        center_text(img, "Text Saved")
        cv.imshow("Original Image", cv.resize(img, (int(scale * WIDTH), int(scale * HEIGHT))))
        cv.waitKey(500)

# Releasing the Capture and Destroying all Windows
cv.destroyAllWindows()