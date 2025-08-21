import cv2
import os

# Input and output folder paths
video_folder = "/home/imad/Research/vid/new_data/drone_dataset"
output_folder = "/home/imad/Research/vid/new_data/drone_dataset_frames"

# VisDrone standard dimensions
frame_width = 1280
frame_height = 720

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate over each video in the video folder
for video_file in sorted(os.listdir(video_folder)):
    if video_file.lower().endswith(".mp4"):
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]  # e.g., "1" from "1.MP4"
        
        # Create a corresponding folder for frames
        frame_folder = os.path.join(output_folder, video_name)
        os.makedirs(frame_folder, exist_ok=True)
        
        # Load the video
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize to VisDrone resolution (960x540)
            frame = cv2.resize(frame, (frame_width, frame_height))

            # Save frame as PNG
            frame_filename = os.path.join(frame_folder, f"{frame_id:05d}.png")
            cv2.imwrite(frame_filename, frame)
            frame_id += 1

        cap.release()
        print(f"✅ Processed {video_file}: {frame_id} frames saved in {frame_folder}")

print("🎉 All videos processed and saved in VisDrone format.")

