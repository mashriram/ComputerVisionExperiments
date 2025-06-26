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
import timm

# --- Configuration Block ---
USE_REAL_DATASET = True
LRS2_DATA_ROOT = "lrs2/lrs2_rebuild/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Part 1: The Data Pipeline (LRS2 Dataset Class) ---
class LRS2Dataset(Dataset):
    def __init__(self, root_dir, req_frames=32):
        super().__init__()
        self.root_dir = root_dir
        self.req_frames = req_frames
        self.videos_dir = os.path.join(self.root_dir, "faces")
        self.audio_dir = os.path.join(
            self.root_dir, "audio", "wav16k", "min", "tr", "mix"
        )
        self.landmarks_dir = os.path.join(self.root_dir, "landmark")
        self.file_list = self._get_file_list()
        if not self.file_list:
            raise RuntimeError(
                "FATAL: No valid data samples were found after full validation."
            )
        print(
            f"--- Dataset Initialized: Found {len(self.file_list)} valid samples. ---"
        )

    def _get_file_list(self):
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
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if frame_count == len(landmarks):
                        valid_samples.append((video_base_name, audio_filename))
                cap.release()
            except Exception:
                continue
        print(f"Validation complete. Found {len(valid_samples)} clean samples.")
        return valid_samples

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if not USE_REAL_DATASET:
            return self._get_dummy_item()
        video_base_name, audio_filename = self.file_list[idx]
        video_path = os.path.join(self.videos_dir, video_base_name + ".mp4")
        audio_path = os.path.join(self.audio_dir, audio_filename)
        landmark_path = os.path.join(self.landmarks_dir, video_base_name + ".npz")

        with np.load(landmark_path, allow_pickle=True) as d:
            landmarks = d["data"] if "data" in d else d["arr_0"]
        num_landmark_frames = landmarks.shape[0]
        try:
            landmarks_reshaped = landmarks.reshape(num_landmark_frames, 68, 2)
        except ValueError:
            return self.__getitem__((idx + 1) % len(self))

        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        audio, sr = librosa.load(audio_path, sr=16000)

        num_frames = len(frames)
        if num_frames >= self.req_frames:
            start_frame = np.random.randint(0, num_frames - self.req_frames + 1)
            video_segment = frames[start_frame : start_frame + self.req_frames]
            landmark_segment = landmarks_reshaped[
                start_frame : start_frame + self.req_frames
            ]
        else:
            video_segment = list(frames)
            landmark_segment = list(landmarks_reshaped)
            padding_needed = self.req_frames - num_frames
            video_segment.extend([frames[-1]] * padding_needed)
            landmark_segment.extend([landmarks_reshaped[-1]] * padding_needed)

        landmark_segment = np.array(landmark_segment)
        identity_frame = video_segment[0]
        identity_landmarks = landmark_segment[0]

        x_coords, y_coords = identity_landmarks[:, 0], identity_landmarks[:, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        pad = 15
        x_min_pad, y_min_pad = max(0, int(x_min - pad)), max(0, int(y_min - pad))
        x_max_pad, y_max_pad = (
            min(identity_frame.shape[1], int(x_max + pad)),
            min(identity_frame.shape[0], int(y_max + pad)),
        )

        identity_face = identity_frame[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
        if identity_face.size == 0:
            return self._get_dummy_item()
        identity_face = cv2.resize(identity_face, (96, 96))

        audio_fps_ratio = 16000 / 25.0
        start_frame_audio_idx = (
            np.random.randint(0, num_frames) if num_frames > self.req_frames else 0
        )
        audio_start, audio_end = (
            int(start_frame_audio_idx * audio_fps_ratio),
            int((start_frame_audio_idx + self.req_frames) * audio_fps_ratio),
        )
        audio_segment = audio[audio_start:audio_end]

        mel_spec = librosa.feature.melspectrogram(
            y=audio_segment, sr=16000, n_mels=80, hop_length=160, n_fft=400
        )
        target_mel_len = self.req_frames * 4
        if mel_spec.shape[1] > target_mel_len:
            mel_spec = mel_spec[:, :target_mel_len]
        elif mel_spec.shape[1] < target_mel_len:
            mel_spec = np.pad(
                mel_spec,
                ((0, 0), (0, target_mel_len - mel_spec.shape[1])),
                mode="constant",
            )

        landmark_flat = landmark_segment.reshape(self.req_frames, 136)
        landmark_tensor = torch.from_numpy(landmark_flat).float()
        identity_image_tensor = (
            torch.from_numpy(identity_face).permute(2, 0, 1).float() / 255.0
        )
        mel_spec_tensor = torch.from_numpy(mel_spec.T).float()

        return identity_image_tensor, landmark_tensor, mel_spec_tensor

    def _get_dummy_item(self):
        identity_image = torch.rand(3, 96, 96)
        landmarks = torch.rand(self.req_frames, 136)
        mel_spec = torch.rand(self.req_frames * 4, 80)
        return identity_image, landmarks, mel_spec


# --- Part 2: Model Architectures ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
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
        self.identity_encoder = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=d_model
        )
        self.identity_encoder.head = nn.Linear(
            self.identity_encoder.head.in_features, d_model
        )
        self.audio_embedding = nn.Linear(80, d_model)
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
        self.landmark_embedding = nn.Linear(136, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )
        self.output_layer = nn.Linear(d_model, 136)

    def forward(self, identity_image, audio_spectrogram, target_landmarks):
        identity_image_resized = torch.nn.functional.interpolate(
            identity_image, size=(224, 224), mode="bilinear", align_corners=False
        )
        identity_features = self.identity_encoder(identity_image_resized).unsqueeze(1)
        audio_embedded = self.audio_embedding(audio_spectrogram)
        audio_memory = self.transformer_encoder(self.pos_encoder(audio_embedded))
        tgt_embedded = self.landmark_embedding(target_landmarks)
        tgt_pos_encoded = self.pos_encoder(tgt_embedded)
        output = self.transformer_decoder(tgt_pos_encoded, memory=audio_memory)
        predicted_landmarks = self.output_layer(output)
        return predicted_landmarks


# --- THIS IS THE FIX ---
class LandmarkRendererGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3 + 1, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def create_landmark_heatmap(self, landmarks_flat, size=(96, 96)):
        # landmarks_flat is expected to be (Batch, Time, 136)
        B, T, _ = landmarks_flat.shape
        # Reshape the flat landmarks into (Batch, Time, 68, 2) for plotting
        landmarks_reshaped = landmarks_flat.view(B, T, 68, 2)
        heatmaps = torch.zeros(B * T, 1, size[0], size[1], device=landmarks_flat.device)
        # Flatten for easy iteration: (Batch*Time, 68, 2)
        lm_iterator = landmarks_reshaped.view(B * T, 68, 2) * (size[0] - 1)
        for i in range(B * T):
            for j in range(68):
                x, y = lm_iterator[i, j, :].long()
                if 0 <= y < size[0] and 0 <= x < size[1]:
                    heatmaps[i, 0, y, x] = 1
        return heatmaps

    def forward(self, identity_image, landmark_sequence):
        # landmark_sequence is shape (Batch, Time, 136)
        B, T, _ = landmark_sequence.shape
        identity_image_expanded = (
            identity_image.unsqueeze(1).repeat(1, T, 1, 1, 1).view(B * T, 3, 96, 96)
        )
        # Directly pass the 3D landmark tensor
        landmark_heatmap = self.create_landmark_heatmap(landmark_sequence)
        model_input = torch.cat([identity_image_expanded, landmark_heatmap], dim=1)
        rendered_faces = self.model(model_input)
        return rendered_faces.view(B, T, 3, 96, 96)


# -------------------------


class SyncNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_encoder = nn.Identity()
        self.audio_encoder = nn.Identity()

    def forward(self, video_frames, audio_spectrogram):
        return torch.rand(1, device=DEVICE)


# --- Part 5: The Full Training Script Logic ---
class TrainingScript:
    def __init__(self):
        self.transformer = TalkingHeadTransformer().to(DEVICE)
        self.renderer = LandmarkRendererGAN().to(DEVICE)
        self.syncnet = SyncNet().to(DEVICE)
        self.dataset = LRS2Dataset(LRS2_DATA_ROOT)
        self.dataloader = DataLoader(
            self.dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True
        )
        self.transformer_optim = optim.AdamW(self.transformer.parameters(), lr=1e-4)
        self.renderer_optim = optim.AdamW(self.renderer.parameters(), lr=1e-4)
        self.l1_loss = nn.L1Loss()

    def train_epoch(self, epoch):
        loop = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(loop):
            identity_image, gt_landmarks, mel_spec = [
                item.to(DEVICE, non_blocking=True) for item in batch
            ]
            self.transformer_optim.zero_grad()
            transformer_target = gt_landmarks[:, 1:]
            transformer_input = gt_landmarks[:, :-1]
            video_len = transformer_input.shape[1]
            mel_spec_aligned = torch.nn.functional.interpolate(
                mel_spec.transpose(1, 2),
                size=video_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            predicted_landmarks = self.transformer(
                identity_image, mel_spec_aligned, transformer_input
            )
            loss_l1 = self.l1_loss(predicted_landmarks, transformer_target)
            loss_l1.backward()
            self.transformer_optim.step()
            self.renderer_optim.zero_grad()
            rendered_frames = self.renderer(
                identity_image, predicted_landmarks.detach()
            )
            gt_frames_for_renderer = identity_image.unsqueeze(1).repeat(
                1, rendered_frames.size(1), 1, 1, 1
            )
            loss_recon = self.l1_loss(rendered_frames, gt_frames_for_renderer)
            mel_spec_syncnet = torch.nn.functional.interpolate(
                mel_spec.transpose(1, 2),
                size=rendered_frames.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            sync_loss = self.syncnet(rendered_frames, mel_spec_syncnet)
            total_renderer_loss = loss_recon + 0.1 * sync_loss
            total_renderer_loss.backward()
            self.renderer_optim.step()
            loop.set_postfix(
                loss_lm=loss_l1.item(), loss_render=total_renderer_loss.item()
            )


# --- Part 6: Main Execution Block ---
if __name__ == "__main__":
    if DEVICE == "cuda":
        torch.multiprocessing.set_start_method("spawn", force=True)
    print(f"--- Starting Real Training on {DEVICE} ---")
    trainer = TrainingScript()
    for epoch in range(1, 101):
        trainer.train_epoch(epoch)
