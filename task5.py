import cv2

def track_object(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: 无法打开 {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\n正在处理 {video_path}...")
    print(">>> 视频正在预览播放，当看到小球时，请按【空格键】暂停！")
    print(">>> 暂停后，用鼠标框选小球，再按【回车键】确认。")

    # 第一阶段：预览视频找小球
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频结束了，没找到球...")
            return
        
        cv2.imshow("Preview (Press SPACE to pause)", frame)
        # 按空格键（ASCII码32）暂停
        if cv2.waitKey(30) & 0xFF == 32:
            break

    cv2.destroyWindow("Preview (Press SPACE to pause)")

    # 第二阶段：暂停后框选
    print(">>> 请画框...")
    bbox = cv2.selectROI("Draw Box and Press ENTER", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Draw Box and Press ENTER")

    # 初始化追踪器
    tracker = cv2.legacy.TrackerCSRT_create()
    tracker.init(frame, bbox)

    print(">>> 开始追踪... 按 'q' 键可提前退出。")

    # 第三阶段：追踪剩余部分
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)
        cv2.imshow("Tracking", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"SUCCESS: 追踪完成，已保存为 {output_path}")

# 运行 example3.mp4
track_object('example3.mp4', '5_track_example3.mp4')

# 运行 example4.mp4
track_object('example4.mp4', '5_track_example4.mp4')