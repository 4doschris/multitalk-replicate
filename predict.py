import os
import torch
from PIL import Image
import torchaudio
import ffmpeg
from multitalk.eval.model_utils import load_model_and_preprocess

MODEL_NAME = "multitalk-hf/multitalk"

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def load_audio(audio_path, target_sr=16000):
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    return waveform.squeeze(0), target_sr

# Load model + processor once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, processor = load_model_and_preprocess(model_name=MODEL_NAME, device=device)

def predict(audio_path: str = "", image_path: str = "", text_input: str = "Describe the content.", max_tokens: int = 256):
    inputs = {}

    if text_input:
        inputs["text"] = text_input

    if image_path and os.path.exists(image_path):
        image = load_image(image_path)
        inputs["image"] = image

    if audio_path and os.path.exists(audio_path):
        audio, sr = load_audio(audio_path)
        inputs["audio"] = audio
        inputs["sampling_rate"] = sr

    # Run model
    response = model.generate_response(
        text=inputs.get("text"),
        image_path=image_path if "image" in inputs else None,
        audio_path=audio_path if "audio" in inputs else None,
        max_new_tokens=max_tokens
    )

    return {"response": response}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="", help="Path to audio file (.wav/.mp3)")
    parser.add_argument("--image_path", type=str, default="", help="Path to image file (.jpg/.png)")
    parser.add_argument("--text_input", type=str, default="Describe the content.", help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens to generate")
    args = parser.parse_args()

    result = predict(
        audio_path=args.audio_path,
        image_path=args.image_path,
        text_input=args.text_input,
        max_tokens=args.max_tokens
    )

    print(result["response"])