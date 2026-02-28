import cv2
import sys
import os
from datetime import date

def video_to_frames(video_path, output_folder):
    # create ouput folder if not already present
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # open video file
    video_capture = cv2.VideoCapture(video_path)
    # get fps to calculate 1/2 second intervals
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    save_interval = int (round(fps/4))
    frame_count = 0
    saved_count = 0
    video_name_f = os.path.basename(video_path)
    video_name = os.path.splitext(video_name_f)[0]
    success = True

    while success:
        # read frames from video
        success, frame = video_capture.read()
        
        if success:
            if frame_count % save_interval == 0:
                # create file name
                file_name = f"{video_name}_frame_{frame_count}.jpg"
                file_path = os.path.join(output_folder, file_name)
            
                # create jpeg
                cv2.imwrite(file_path, frame)

                saved_count += 1
            
            frame_count += 1
    # release
    video_capture.release()
    print(f"Created {saved_count} images at {output_folder}.")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        video_to_frames(sys.argv[1], sys.argv[2])