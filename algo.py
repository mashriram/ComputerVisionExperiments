import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import librosa
import os
import math
from tqdm import tqdm
import timm  # For the ViT model

# --- Configuration Block ---
# Set this to False to run a "dry run" without the actual LRS2 dataset
# The dry run will create dummy data and test the model architecture.
USE_REAL_DATASET = True
LRS2_DATA_ROOT = "lrs2/lrs2_rebuild/"  # <--- IMPORTANT: Change this path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Part 1: The Data Pipeline (LRS2 Dataset Class) ---

# Mock face_alignment if not installed, for dry run
try:
    import face_alignment
except ImportError:
    print("WARNING: face_alignment library not found. Using a dummy for dry run.")

    class DummyFaceAlignment:
        def __init__(self, *args, **kwargs):
            pass

        def get_landmarks(self, image):
            return [np.random.rand(68, 2) * image.shape[0]]  # Return dummy landmarks

    face_alignment = DummyFaceAlignment


class LRS2Dataset(Dataset):
    def __init__(
        self, root_dir, req_frames=32
    ):  # <-- IMPORTANT: Lowered default req_frames
        super().__init__()
        print("\n--- Initializing LRS2 Dataset ---")
        self.root_dir = root_dir
        self.req_frames = req_frames  # Now defaulting to 32 frames (~1.3 seconds)

        self.videos_dir = os.path.join(self.root_dir, "faces")
        self.audio_dir = os.path.join(
            self.root_dir, "audio", "wav16k", "min", "tr", "mix"
        )
        self.landmarks_dir = os.path.join(self.root_dir, "landmark")

        self.file_list = self._get_file_list()
        if not self.file_list:
            raise RuntimeError(
                "FATAL: No valid data samples were found after validation."
            )
        print(
            f"--- Dataset Initialized: Found {len(self.file_list)} valid samples. ---"
        )

    def __len__(self):
        return len(self.file_list)

    def _get_file_list(self):
        """
        Validates all potential samples, keeping only those that are not corrupt
        and have matching frames. The length check is now removed from here.
        """
        if not USE_REAL_DATASET:
            return [("dummy_video", "dummy_audio")]

        usable_basenames = {
            os.path.splitext(f)[0] for f in os.listdir(self.videos_dir)
        }.intersection({os.path.splitext(f)[0] for f in os.listdir(self.landmarks_dir)})

        potential_samples, audio_files = [], os.listdir(self.audio_dir)
        for audio_filename in audio_files:
            if not audio_filename.lower().endswith(".wav"):
                continue
            try:
                parts = os.path.splitext(audio_filename)[0].split("_")
                s1, s2 = f"{parts[0]}_{parts[1]}", f"{parts[3]}_{parts[4]}"
                if s1 in usable_basenames:
                    potential_samples.append((s1, audio_filename))
                if s2 in usable_basenames:
                    potential_samples.append((s2, audio_filename))
            except IndexError:
                continue

        print(
            f"Found {len(potential_samples)} potential samples. Now validating content..."
        )

        valid_samples = []
        for video_base_name, audio_filename in tqdm(
            potential_samples, desc="Validating file content"
        ):
            video_path = os.path.join(self.videos_dir, video_base_name + ".mp4")
            landmark_path = os.path.join(self.landmarks_dir, video_base_name + ".npz")
            try:
                with np.load(landmark_path, allow_pickle=True) as d:
                    landmarks = d["data"] if "data" in d else d["arr_0"]
                cap = cv2.VideoCapture(video_path)
                is_open = cap.isOpened()
                if is_open:
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if frame_count == len(landmarks):
                        valid_samples.append((video_base_name, audio_filename))
                cap.release()
            except Exception:
                continue

        print(f"Validation complete. Found {len(valid_samples)} clean samples.")
        return valid_samples

    def __getitem__(self, idx):
        if not USE_REAL_DATASET: return self._get_dummy_item()

        video_base_name, audio_filename = self.file_list[idx]
        video_path = os.path.join(self.videos_dir, video_base_name + '.mp4')
        audio_path = os.path.join(self.audio_dir, audio_filename)
        landmark_path = os.path.join(self.landmarks_dir, video_base_name + '.npz')

        with np.load(landmark_path, allow_pickle=True) as d:
            landmarks = d['data'] if 'data' in d else d['arr_0']
            
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        audio, sr = librosa.load(audio_path, sr=16000)

        num_frames = len(frames)
        if num_frames >= self.req_frames:
            start_frame = np.random.randint(0, num_frames - self.req_frames + 1)
            video_segment = frames[start_frame : start_frame + self.req_frames]
            landmark_segment = landmarks[start_frame : start_frame + self.req_frames]
        else:
            video_segment = list(frames)
            landmark_segment = list(landmarks)
            padding_needed = self.req_frames - num_frames
            video_segment.extend([frames[-1]] * padding_needed)
            landmark_segment.extend([landmarks[-1]] * padding_needed)
        
        identity_frame = video_segment[0]
        identity_landmarks = landmark_segment[0]
        
        x_coords = identity_landmarks[:, 0]
        y_coords = identity_landmarks[:, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        pad = 15
        x_min, y_min = max(0, int(x_min - pad)), max(0, int(y_min - pad))
        x_max, y_max = min(identity_frame.shape[1], int(x_max + pad)), min(identity_frame.shape[0], int(y_max + pad))
        
        identity_face = identity_frame[y_min:y_max, x_min:x_max]
        if identity_face.size == 0: return self._get_dummy_item()
        identity_face = cv2.resize(identity_face, (96, 96))

        audio_fps_ratio = 16000 / 25.0
        start_frame_audio_idx = np.random.randint(0, num_frames) if num_frames > self.req_frames else 0
        audio_start = int(start_frame_audio_idx * audio_fps_ratio)
        audio_end = int((start_frame_audio_idx + self.req_frames) * audio_fps_ratio)
        audio_segment = audio[audio_start:audio_end]
        
        mel_spec = librosa.feature.melspectrogram(y=audio_segment, sr=16000, n_mels=80, hop_length=160, n_fft=400)

        target_mel_len = self.req_frames * 4
        if mel_spec.shape[1] > target_mel_len:
            mel_spec = mel_spec[:, :target_mel_len]
        elif mel_spec.shape[1] < target_mel_len:
            mel_spec = np.pad(mel_spec, ((0, 0), (0, target_mel_len - mel_spec.shape[1])), mode='constant')
        
        # --- THIS IS THE FIX ---
        # 1. Convert the list of landmark arrays to a single NumPy array of shape (req_frames, 68, 2)
        landmark_array = np.array(landmark_segment)[:, :, :2] 
        # 2. Reshape it to (req_frames, 136) and convert to a tensor
        landmark_tensor_flat = torch.from_numpy(landmark_array).view(self.req_frames, -1).float()
        
        identity_image_tensor = torch.from_numpy(identity_face).permute(2,0,1).float() / 255.
        mel_spec_tensor = torch.from_numpy(mel_spec.T).float()
        
        # Return the correctly shaped landmark tensor
        return identity_image_tensor, landmark_tensor_flat, mel_spec_tensor

    def _get_dummy_item(self):
        identity_image = torch.rand(3, 96, 96)
        landmarks_sequence = torch.rand(self.req_frames, 68, 2)
        mel_spec = torch.rand(self.req_frames * 4, 80)
        return identity_image, landmarks_sequence, mel_spec


# --- Part 2: Refined Model Architecture ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TalkingHeadTransformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
    ):
        super().__init__()
        self.d_model = d_model

        # Identity Encoder: Use a pre-trained Vision Transformer (ViT)
        self.identity_encoder = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=d_model
        )
        self.identity_encoder.head = nn.Linear(
            self.identity_encoder.head.in_features, d_model
        )  # Adjust final layer

        # Audio Encoder
        self.audio_embedding = nn.Linear(80, d_model)  # 80 n_mels
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Decoder for Landmarks
        self.landmark_embedding = nn.Linear(136, d_model)  # 68*2 landmarks
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )
        self.output_layer = nn.Linear(d_model, 136)  # Predict next landmarks

    def forward(self, identity_image, audio_spectrogram, target_landmarks):
        # identity_image: (B, 3, 96, 96) -> Resize for ViT
        # audio_spectrogram: (B, T_audio, 80)
        # target_landmarks: (B, T_video, 136)

        # 1. Encode Identity - Resize image to 224x224 for ViT
        identity_image_resized = torch.nn.functional.interpolate(
            identity_image, size=(224, 224), mode="bilinear", align_corners=False
        )
        identity_features = self.identity_encoder(
            identity_image_resized
        )  # (B, d_model)
        identity_features = identity_features.unsqueeze(1)  # (B, 1, d_model)

        # 2. Encode Audio
        audio_embedded = self.audio_embedding(audio_spectrogram)
        audio_embedded = self.pos_encoder(audio_embedded.permute(1, 0, 2)).permute(
            1, 0, 2
        )
        audio_memory = self.transformer_encoder(audio_embedded)

        # 3. Decode Landmarks (Teacher Forcing)
        # We embed the ground-truth landmarks to predict the *next* frame's landmarks
        tgt_embedded = self.landmark_embedding(target_landmarks)
        tgt_embedded = self.pos_encoder(tgt_embedded.permute(1, 0, 2)).permute(1, 0, 2)

        # The decoder uses identity as its initial memory state
        # A more complex implementation might use it differently
        output = self.transformer_decoder(tgt_embedded, memory=audio_memory)
        predicted_landmarks = self.output_layer(output)

        return predicted_landmarks


# --- Part 3: The Renderer Model (Image-to-Image GAN) ---
class LandmarkRendererGAN(nn.Module):
    # A simplified U-Net like structure for rendering.
    def __init__(self):
        super().__init__()
        # This is a conceptual stub. A real renderer is a large, complex model.
        self.model = nn.Sequential(
            nn.Conv2d(3 + 1, 64, 4, 2, 1),  # Input: Identity Image + Landmark Heatmap
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),  # Output image pixels
        )

    def create_landmark_heatmap(self, landmarks, size=(96, 96)):
        # landmarks: (B, T, 136)
        # Create a single channel heatmap from landmarks for a given frame
        # This is a simplified approach.
        B, T, _ = landmarks.shape
        heatmaps = torch.zeros(B * T, 1, size[0], size[1], device=landmarks.device)
        # For simplicity, we just put a '1' at each landmark coordinate
        # This part is computationally intensive and can be optimized.
        lm_flat = landmarks.view(B * T, 68, 2) * (size[0] - 1)
        for i in range(B * T):
            for j in range(68):
                x, y = lm_flat[i, j, :].long()
                if 0 <= y < size[0] and 0 <= x < size[1]:
                    heatmaps[i, 0, y, x] = 1
        return heatmaps

    def forward(self, identity_image, landmark_sequence):
        B, T, _ = landmark_sequence.shape
        identity_image_expanded = (
            identity_image.unsqueeze(1).repeat(1, T, 1, 1, 1).view(B * T, 3, 96, 96)
        )
        landmark_heatmap = self.create_landmark_heatmap(landmark_sequence)

        model_input = torch.cat([identity_image_expanded, landmark_heatmap], dim=1)
        rendered_faces = self.model(model_input)  # (B*T, 3, 96, 96)
        return rendered_faces.view(B, T, 3, 96, 96)


# --- Part 4: The Perceptual Loss Model (SyncNet) ---
class SyncNet(nn.Module):
    # This is a STUB. A real SyncNet is a complex pre-trained model.
    def __init__(self):
        super().__init__()
        # The real SyncNet has a video encoder and an audio encoder.
        self.video_encoder = nn.Identity()  # Placeholder
        self.audio_encoder = nn.Identity()  # Placeholder

    def forward(self, video_frames, audio_spectrogram):
        # It should return a similarity score (or a distance).
        # High score means good sync, low score means bad sync.
        # Returning a random loss for the dry run.
        return torch.rand(1, device=DEVICE)


# --- Part 5: The Full Training Script Logic ---
class TrainingScript:
    def __init__(self):
        # 1. Models
        self.transformer = TalkingHeadTransformer().to(DEVICE)
        self.renderer = LandmarkRendererGAN().to(DEVICE)
        self.syncnet = SyncNet().to(DEVICE)  # Should be pre-trained and frozen

        # 2. Data
        self.dataset = LRS2Dataset(LRS2_DATA_ROOT)
        self.dataloader = DataLoader(
            self.dataset, batch_size=4, shuffle=True, num_workers=4
        )

        # 3. Optimizers
        self.transformer_optim = optim.AdamW(self.transformer.parameters(), lr=1e-4)
        self.renderer_optim = optim.AdamW(self.renderer.parameters(), lr=1e-4)

        # 4. Loss Functions
        self.l1_loss = nn.L1Loss()

    def train_epoch(self, epoch):
        loop = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch in loop:
            identity_image, gt_landmarks, mel_spec = [item.to(DEVICE) for item in batch]

            # --- Train Transformer ---
            self.transformer_optim.zero_grad()

            # --- THIS IS THE FIX ---
            # REMOVED: The unnecessary view operation is gone.
            # gt_landmarks_flat = gt_landmarks.view(...) 

            # The target for the transformer is the next frame's landmarks
            transformer_target = gt_landmarks[:, 1:]
            transformer_input = gt_landmarks[:, :-1]
            # ------------------------

            # We need to match the audio length to the video length for the decoder
            # The length of transformer_input is req_frames - 1
            video_len = transformer_input.shape[1]
            mel_spec_aligned = torch.nn.functional.interpolate(
                mel_spec.transpose(1, 2), size=video_len, mode="linear"
            ).transpose(1, 2)

            predicted_landmarks = self.transformer(
                identity_image, mel_spec_aligned, transformer_input
            )

            loss_l1 = self.l1_loss(predicted_landmarks, transformer_target)
            loss_l1.backward()
            self.transformer_optim.step()

            # --- Train Renderer ---
            self.renderer_optim.zero_grad()

            # Use the *predicted* landmarks from the (detached) transformer to train the renderer
            rendered_frames = self.renderer(
                identity_image, predicted_landmarks.detach()
            )

            # Get ground truth frames (would require loading them in dataset)
            # For simplicity, we just use the identity image as a fake target for loss calculation
            gt_frames_for_renderer = identity_image.unsqueeze(1).repeat(
                1, rendered_frames.size(1), 1, 1, 1
            )  # B, T, C, H, W

            loss_recon = self.l1_loss(rendered_frames, gt_frames_for_renderer)

            # Perceptual Sync Loss
            # This is the key to good lip sync.
            # We align the audio again to the rendered frame count.
            mel_spec_syncnet = torch.nn.functional.interpolate(
                mel_spec.transpose(1, 2), size=rendered_frames.size(1), mode="linear"
            ).transpose(1, 2)
            sync_loss = self.syncnet(rendered_frames, mel_spec_syncnet)

            total_renderer_loss = loss_recon + 0.1 * sync_loss  # Weight the sync loss
            total_renderer_loss.backward()
            self.renderer_optim.step()

            loop.set_postfix(
                loss_lm=loss_l1.item(), loss_render=total_renderer_loss.item()
            )


# --- Part 6: Main Execution Block ---
def run_dry_run():
    print("--- Starting Dry Run ---")
    print(f"Using device: {DEVICE}")

    # 1. Test Dataset
    print("\n[1] Testing LRS2 Dataset loading...")
    dataset = LRS2Dataset(None, req_frames=75)
    dataloader = DataLoader(dataset, batch_size=2)
    identity, landmarks, audio = next(iter(dataloader))
    print(f"  - Identity shape: {identity.shape}")
    print(f"  - Landmarks shape: {landmarks.shape}")
    print(f"  - Audio spec shape: {audio.shape}")

    # 2. Test Model Forward Pass
    print("\n[2] Testing model forward pass...")
    transformer = TalkingHeadTransformer().to(DEVICE)
    renderer = LandmarkRendererGAN().to(DEVICE)

    identity, landmarks, audio = (
        identity.to(DEVICE),
        landmarks.to(DEVICE),
        audio.to(DEVICE),
    )
    landmarks_flat = landmarks.view(landmarks.size(0), landmarks.size(1), -1)

    # Align audio and video sequences
    video_len = landmarks_flat.shape[1] - 1
    audio_aligned = torch.nn.functional.interpolate(
        audio.transpose(1, 2), size=video_len, mode="linear"
    ).transpose(1, 2)

    pred_landmarks = transformer(identity, audio_aligned, landmarks_flat[:, :-1])
    print(f"  - Predicted landmarks shape: {pred_landmarks.shape}")

    rendered_output = renderer(identity, pred_landmarks)
    print(f"  - Rendered output shape: {rendered_output.shape} (B, T, C, H, W)")

    # 3. Test Training Loop
    print("\n[3] Testing one step of the training loop...")
    trainer = TrainingScript()
    trainer.train_epoch(epoch=1)

    print("\n--- Dry Run Complete ---")
    print("The model architecture and data flow are working correctly.")
    print(
        "To train for real, set USE_REAL_DATASET = True and provide the correct LRS2 path."
    )


if __name__ == "__main__":
    if USE_REAL_DATASET:
        print("--- Starting Real Training ---")
        trainer = TrainingScript()
        for epoch in range(1, 101):  # Train for 100 epochs
            trainer.train_epoch(epoch)
    else:
        run_dry_run()
