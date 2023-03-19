import cv2
import numpy as np
from draw_bb.drawbb import draw_bb


#load video
cap = cv2.VideoCapture('drive/MyDrive/MIDS/w251/final_project/desk.mp4')

# Check if the VideoCapture object was successfully opened
if not cap.isOpened():
    print("Error: Could not open video file")
    
def video_processor(vid):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        #run model
        # bb_boxes = model.run()

        # Display the frame
        for bb in bb_boxes:
          frame = draw_bb(bb,frame)
    
        print("picture: " + str(i))
        cv2_imshow(frame)
        cv2.destroyAllWindows()

        cv2.waitKey(1)

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
