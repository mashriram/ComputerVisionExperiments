import torch
import numpy as np
import cv2
import librosa
import os
import face_alignment
import pyttsx3
from tqdm import tqdm  # <-- New TTS library

# Import your model classes from the training script
# Make sure your training script is named 'algo1.py'
from algo1 import TalkingHeadTransformer, LandmarkRendererGAN

# --- Configuration ---
CHECKPOINT_PATH = "checkpoint.pth"
INPUT_IMAGE_PATH = "person.jpg"  # Make sure this image exists
INPUT_TEXT = "The quick brown fox jumps over the lazy dog."
OUTPUT_VIDEO_PATH = "generated_video.mp4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_audio_with_pyttsx3(text, output_path):
    """
    Generates an audio file from text using the pyttsx3 library.
    """
    engine = pyttsx3.init()
    # Optional: You can set properties like rate and voice
    # rate = engine.getProperty('rate')
    # engine.setProperty('rate', rate - 50)
    engine.save_to_file(text, output_path)
    # This is crucial: It processes the command queue and saves the file.
    engine.runAndWait()
    print(f"Audio successfully generated and saved to {output_path}")


def main():
    print("--- Starting Inference ---")

    # 1. Load Models
    print("Loading models...")
    transformer = TalkingHeadTransformer().to(DEVICE).eval()
    renderer = LandmarkRendererGAN().to(DEVICE).eval()

    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint file not found at {CHECKPOINT_PATH}. Please train the model first."
        )

    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    transformer.load_state_dict(checkpoint["transformer_state_dict"])
    renderer.load_state_dict(checkpoint["renderer_state_dict"])
    print("Models loaded successfully.")

    # 2. Prepare Inputs
    # --- NEW TTS SECTION ---
    print("Generating audio with pyttsx3...")
    audio_path = "temp_audio.wav"
    generate_audio_with_pyttsx3(INPUT_TEXT, audio_path)
    # -----------------------

    # Preprocess Audio
    audio, sr = librosa.load(audio_path, sr=16000)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=16000, n_mels=80, hop_length=160, n_fft=400
    )
    mel_spec_tensor = torch.from_numpy(mel_spec.T).unsqueeze(0).to(DEVICE)

    # Preprocess Image
    print("Processing input image...")
    if not os.path.isfile(INPUT_IMAGE_PATH):
        raise FileNotFoundError(
            f"Input image not found at {INPUT_IMAGE_PATH}. Please provide a valid path."
        )

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=DEVICE)
    input_image_full = cv2.imread(INPUT_IMAGE_PATH)
    preds = fa.get_landmarks(input_image_full)
    if preds is None:
        raise ValueError("No face detected in the input image.")

    identity_landmarks = preds[0]
    x_coords, y_coords = identity_landmarks[:, 0], identity_landmarks[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    pad = 15
    x_min_pad, y_min_pad = max(0, int(x_min - pad)), max(0, int(y_min - pad))
    x_max_pad, y_max_pad = (
        min(input_image_full.shape[1], int(x_max + pad)),
        min(input_image_full.shape[0], int(y_max + pad)),
    )

    identity_face = input_image_full[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
    identity_face_resized = cv2.resize(identity_face, (96, 96))
    identity_image_tensor = (
        torch.from_numpy(identity_face_resized)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        .to(DEVICE)
        / 255.0
    )

    # 3. Autoregressive Generation
    print("Generating landmark sequence...")
    # Determine number of frames based on audio length at 25 fps
    num_frames = int(len(audio) / sr * 25)

    # Start with the initial landmarks from the image, flattened to (1, 136)
    initial_landmarks_flat = (
        torch.from_numpy(identity_landmarks[:, :2].reshape(1, 136)).float().to(DEVICE)
    )
    current_landmarks = initial_landmarks_flat.clone().unsqueeze(
        1
    )  # Shape: (1, 1, 136)

    generated_landmark_sequence = []

    # Align audio to the number of frames to generate
    mel_spec_aligned = torch.nn.functional.interpolate(
        mel_spec_tensor.transpose(1, 2), size=num_frames, mode="linear"
    ).transpose(1, 2)

    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="Generating frames"):
            # The model predicts the *next* frame based on the current one
            prediction = transformer(
                identity_image_tensor,
                mel_spec_aligned[:, i : i + 1, :],
                current_landmarks,
            )
            generated_landmark_sequence.append(prediction)
            # For the next step, use the prediction as the new input
            current_landmarks = prediction

    full_landmark_sequence = torch.cat(generated_landmark_sequence, dim=1)

    # 4. Render the video
    print("Rendering video from landmarks...")
    with torch.no_grad():
        rendered_frames = renderer(identity_image_tensor, full_landmark_sequence)

    # 5. Save the output video
    print(f"Saving video to {OUTPUT_VIDEO_PATH}...")
    rendered_frames_np = (
        rendered_frames.squeeze(0).permute(0, 2, 3, 1).cpu().numpy() * 255
    ).astype(np.uint8)

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), 25, (96, 96)
    )
    for frame in rendered_frames_np:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    print("--- Inference Complete ---")
    print(
        f"Video saved to {OUTPUT_VIDEO_PATH}. You can now combine it with '{audio_path}' using a video editor or ffmpeg."
    )


if __name__ == "__main__":
    main()
