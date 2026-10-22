
import cv2
import numpy as np

img = cv2.imread('example1.jpg')

if img is None:
    print("ERROR: Cannot find example1.jpg")
else:
    # 1. 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 高斯模糊（去除噪点，让边缘更平滑）
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Canny 边缘检测（找像素突变的边界）
    edges = cv2.Canny(blurred, 50, 150)
    
    # 4. 形态学膨胀（把断断续续的边缘线连成一个完整的圈）
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # 5. 找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找出面积最大的轮廓（只圈飞机，过滤掉背景噪音）
        largest_contour = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # 画红色的轮廓线
        cv2.drawContours(img, [largest_contour], -1, (0, 0, 255), 3)
        
        print(f"Total Area: {area}")
        print(f"Total Perimeter: {perimeter}")

    # 保存并显示
    cv2.imwrite('3_contours_final.jpg', img)
    print("SUCCESS: Saved as 3_contours_final.jpg")

    cv2.imshow('Final Contours (Press any key)', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()