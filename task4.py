import cv2
import numpy as np

cap = cv2.VideoCapture('example2.mp4')
if not cap.isOpened():
    print("ERROR: 找不到 example2.mp4，请确认它在文件夹里！")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter('4_mog2_output.mp4', fourcc, fps, (width * 2, height))

mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

print("正在处理视频... 按 'q' 键可以提前退出。")

while True:
    ret, frame = cap.read()
    if not ret:
        break 

    left_frame = frame.copy()
    cv2.putText(left_frame, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    fg_mask = mog2.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
    
    right_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
    cv2.putText(right_frame, "MOG2 Subtraction", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    combined = np.hstack((left_frame, right_frame))
    out.write(combined)

    display = cv2.resize(combined, (1200, int(1200 * height / (width * 2))))
    cv2.imshow('Comparison (Press Q to quit)', display)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("SUCCESS: 视频处理完成，已保存为 4_mog2_output.mp4")