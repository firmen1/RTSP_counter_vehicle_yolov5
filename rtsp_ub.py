from deep_sort_realtime.deepsort_tracker import DeepSort
from yoloDetector import YoloDetector
from preprocessingTracker import PreprocessingTracker
import cv2
import time


cap = cv2.VideoCapture("rtsp://10.45.188.254:31554/nvstream/home/vst/vst_release/streamer_videos/veteran_in_20230522T111503.mkv")
object_tracker = DeepSort()

model_1 = 'C:/Users/acer/Documents/coolyeah/skripsi/Revisi/real time code/no_preprocessing_model_skripsi.pt'
model_2 = 'C:/Users/acer/Documents/coolyeah/skripsi/Revisi/real time code/preprocessing_model_skripsi.pt'

detector = YoloDetector(model_2)
preprocessing_frame = PreprocessingTracker()

preprocessing_type = 'machine_learning'
best_confidence = 0

set_id_datang = set()
set_id_lewat = set()
# Define the counting line
line_start = (0, 466)
line_end = (1100, 466)
width  = 1280
length = 720
# Initialize counters
counter_car = 0
counter_motorbike = 0
while True or cap.isOpened():
    start = time.perf_counter()

    success, img = cap.read()
    if not success:
        break
    if preprocessing_type == 'machine_learning':
        preprocessing_img = preprocessing_frame.learning_preprocessing(img)
        best_confidence = 0.421
    elif preprocessing_type == 'reduce_glare':
        preprocessing_img = preprocessing_frame.mix_filter(img)
    else:
        preprocessing_img = img
        best_confidence = 0.566
    # preprocessing_img = img
    # best_confidence = 0.566
    # img = cv2.resize(img, (width, length))
    
    results = detector.score_frame(preprocessing_img)
    img, detections = detector.plot_boxes(results, preprocessing_img, height=preprocessing_img.shape[0], width=preprocessing_img.shape[1], confidence=best_confidence)
    tracks = object_tracker.update_tracks(detections, frame=img)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()

        bbox = ltrb
        # c_x, c_y, _, _ =  track.to_tlbr()

        # Draw the counting line
        cv2.line(img, line_start, line_end, (255, 250, 250), 2)

        # Draw bounding box
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

        # Draw track ID and class label
        cv2.putText(img, f"ID: {str(track_id)} {track.get_det_class()}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # Check if the bounding box crosses the counting line
        if bbox[1] < line_start[1] and bbox[3] > line_start[1]:
            if track.get_det_class() == 'Car' and track_id not in set_id_lewat and track_id in set_id_datang:
                counter_car += 1
                set_id_lewat.add(track_id)
            elif track.get_det_class() == 'Motorbike' and track_id not in set_id_lewat and track_id in set_id_datang:
                counter_motorbike += 1
                set_id_lewat.add(track_id)
        else:
            set_id_datang.add(track_id) if track_id not in set_id_datang else None

    end = time.perf_counter()
    totalTime = end - start
    fps = 1 / totalTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 720 - 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Display counters
    cv2.putText(img, f'Car Count: {counter_car}', (20, 720 - 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.putText(img, f'Motorbike Count: {counter_motorbike}', (20, 720 - 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow("Object Counting", img)
    time.sleep(0.1)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
