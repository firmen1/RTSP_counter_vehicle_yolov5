import cv2
import time

video =cv2.VideoCapture("rtsp://10.45.188.254:31554/nvstream/home/vst/vst_release/streamer_videos/veteran_in_20230522T111503.mkv")

while True:
    start = time.perf_counter()
    _, frame = video.read()
    img = cv2.resize(frame, (1280, 720))
    end = time.perf_counter()
    totalTime = end - start
    fps = 1 / totalTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 720 - 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.imshow('RTSP', img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()