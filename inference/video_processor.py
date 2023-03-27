import cv2
import numpy as np
from draw_bb.drawbb import draw_bb

from casapose.inf_casapose import runnetwork

# Load model & weights
#model = cv2.dnn.readNetFromTensorflow("frozen_model.pb")

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
        _, estimated_points, estimated_poses, output_seg = runnetwork(frame)
        #bb_boxes = model.run()
        # model.setInput(frame)

        # output = model.forward()

        # Display the frame
        frame = draw_bb(estimated_points,frame) # add in output
    
        print("picture: " + str(i))
        cv2.imshow(frame)
        cv2.destroyAllWindows()

        cv2.waitKey(1)

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
