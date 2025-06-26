import os
import cv2
from tqdm import tqdm

# --- Configuration ---
# The path to the folder containing your MP4 files
VIDEOS_DIR = "lrs2/lrs2_rebuild/faces"

# A new folder where we will move the bad files
QUARANTINE_DIR = "lrs2/lrs2_rebuild/bad_videos"

def clean_video_dataset():
    """
    Checks every video file in a directory for corruption and moves bad
    files to a quarantine folder.
    """
    print(f"--- Starting Dataset Cleanup ---")
    print(f"Video source: '{VIDEOS_DIR}'")
    print(f"Quarantine for bad files: '{QUARANTINE_DIR}'")

    if not os.path.isdir(VIDEOS_DIR):
        print(f"FATAL: Video directory not found. Exiting.")
        return

    # Create the quarantine directory if it doesn't exist
    os.makedirs(QUARANTINE_DIR, exist_ok=True)

    video_files = os.listdir(VIDEOS_DIR)
    bad_files_count = 0

    for filename in tqdm(video_files, desc="Checking videos"):
        if not filename.endswith('.mp4'):
            continue
        
        video_path = os.path.join(VIDEOS_DIR, filename)
        
        # Try to open the video file
        cap = cv2.VideoCapture(video_path)
        
        # If it's not opened, the file is corrupt
        if not cap.isOpened():
            bad_files_count += 1
            # Move the bad file to the quarantine folder
            destination_path = os.path.join(QUARANTINE_DIR, filename)
            try:
                os.rename(video_path, destination_path)
            except OSError as e:
                print(f"\nCould not move file {filename}. Error: {e}")

        # Release the capture object
        cap.release()

    print(f"\n--- Cleanup Complete ---")
    print(f"Found and moved {bad_files_count} corrupt video files to '{QUARANTINE_DIR}'.")
    print("Your 'faces' directory is now clean.")

if __name__ == '__main__':
    clean_video_dataset()