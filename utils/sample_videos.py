import os
import cv2
import random
from tqdm import tqdm  # Import tqdm to create the progress bar

def sample_videos(directory_path, output_path, num_frames_per_recording):
    # Get the list of all files in the directory
    all_files = os.listdir(directory_path)

    # Filter out all .mp4 files
    video_files = [file for file in all_files if file.endswith(".mp4")]

    # Create a tqdm progress bar with the total number of videos to process
    progress_bar = tqdm(video_files, desc="Processing videos", unit="video", ncols=100)

    # For each video file
    for video_file in progress_bar:
        video_path = os.path.join(directory_path, video_file)
        vidcap = cv2.VideoCapture(video_path)

        # Get the total number of frames
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= num_frames_per_recording:
            print(f"Warning: Video '{video_path}' has fewer than {num_frames_per_recording} frames. Skipping...")
            continue

        # Generate 5 random frame numbers
        start_from_frame = int(0.95*total_frames)
        frame_numbers = random.sample(range(start_from_frame, total_frames), num_frames_per_recording)

        # Sort frame numbers (not necessary but could speed up the process)
        frame_numbers.sort()

        for frame_number in frame_numbers:
            # Set the current frame position
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read the current frame
            success, image = vidcap.read()

            if success and image.size > 0:
                # Write the current frame to a .jpg file
                cv2.imwrite(f"{output_path}/{video_file}_frame_{frame_number}.jpg", image)
            else:
                print(f"Warning: Empty or corrupted frame {frame_number} in {video_file}")

        # Close the video file
        vidcap.release()

if __name__ == '__main__':
    directory_path = "../video/old/"
    output_path = "../data/yolov5/recording_frames_2nd_sample"
    num_frames_per_recording = 2
    sample_videos(directory_path, output_path, num_frames_per_recording)
