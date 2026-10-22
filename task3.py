
import cv2
import numpy as np

img = cv2.imread('example1.jpg')

if img is None:
    print("ERROR: Cannot find example1.jpg")
else:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
    
        cv2.drawContours(img, [largest_contour], -1, (0, 0, 255), 3)
        
        print(f"Total Area: {area}")
        print(f"Total Perimeter: {perimeter}")

    cv2.imwrite('3_contours_final.jpg', img)
    print("SUCCESS: Saved as 3_contours_final.jpg")

    cv2.imshow('Final Contours (Press any key)', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()