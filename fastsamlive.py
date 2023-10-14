from fastsam import FastSAM, FastSAMPrompt
import torch
import numpy as np
import cv2
import time
import apriltag

model = FastSAM('FastSAM-x.pt')


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(DEVICE)

cap = cv2.VideoCapture(4)
desired_width = 1280
desired_height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

roi_x, roi_y, roi_w, roi_h = 600, 240, 400, 240

detector = apriltag.Detector()

while cap.isOpened():
    
    suc, frame = cap.read()

    if not suc:
        print("Ignoring empty camera frame.")
        continue

    frame_roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    gray_frame = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray_frame)


    start = time.perf_counter()
    
    everything_results = model(
        source=frame_roi,
        device=DEVICE,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
    )
    
    end = time.perf_counter()
    total_time = end - start
    fps = 1 / total_time
    
    if everything_results is not None:
        for box in everything_results[0].boxes:
            box = box.xyxy.cpu().numpy()[0]
            cv2.rectangle(frame_roi, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

        if len(detections) > 0:
            for i, tag in enumerate(detections):
                tag_id = tag.tag_id
                corners = tag.corners.astype(int)
                center_x = int(tag.center[0])
                center_y = int(tag.center[1])
                # print(f"Tag {tag_id} Center Pixel Coordinates: ({center_x}, {center_y})")
                # print(f"Tag {tag_id} Corner Pixel Coordinates: LU{str(corners[0])}, RD{str(corners[2])}")
                # print("---------------------------------")
                cv2.polylines(frame_roi, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.circle(frame_roi, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)
                cv2.putText(frame_roi, f"Tag {tag_id}", (center_x - 20, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
        prompt_process = FastSAMPrompt(frame_roi, everything_results, device=DEVICE)
        ann = prompt_process.everything_prompt()
        img = prompt_process.plot_to_result(annotations=ann, withContours=True, better_quality=True, retina=False)    
        cv2.putText(img, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('img', img)
        print('-'*30)
        print(type(ann))
        print('-'*30)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()
