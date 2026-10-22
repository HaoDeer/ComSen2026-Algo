import cv2
import numpy as np
from PIL import Image

img = cv2.imread('test.jpg')

if img is None:
    print("ERROR: Cannot find test.jpg")
else:
    print("SUCCESS: Image loaded! Shape:", img.shape)
    pil_img = Image.open('test.jpg')
    print("SUCCESS: Pillow loaded, format:", pil_img.format)

    b, g, r = cv2.split(img)
    cv2.imwrite('1_channel_blue.jpg', b)
    cv2.imwrite('1_channel_green.jpg', g)
    cv2.imwrite('1_channel_red.jpg', r)
    print("SUCCESS: Channels split and saved!")

    cv2.imshow('Original (Press any key)', img)
    cv2.imshow('Blue Channel', b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rows, cols = img.shape[:2]
    resized = cv2.resize(img, (400, 300))
    cv2.imwrite('2_resized.jpg', resized)

    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite('2_rotated.jpg', rotated)

    M_translate = np.float32([[1, 0, 100], [0, 1, 50]])
    translated = cv2.warpAffine(img, M_translate, (cols, rows))
    cv2.imwrite('2_translated.jpg', translated)

    flipped = cv2.flip(img, 1)
    cv2.imwrite('2_flipped.jpg', flipped)
    
    print("ALL DONE! Check the new images in the folder.")