"""Wan2.1 Image-to-Video API for Salad deployment."""
import io
import os
import uuid
from pathlib import Path

import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

OUT_DIR = Path("/tmp/videos")
OUT_DIR.mkdir(exist_ok=True)

app = FastAPI()
pipe: WanImageToVideoPipeline | None = None

MODEL_ID = os.getenv("MODEL_ID", "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")


@app.on_event("startup")
async def load_model():
    global pipe
    print(f"Loading {MODEL_ID}...")
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.enable_model_cpu_offload()
    print("Model ready.")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "ready": pipe is not None}


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    prompt: str = Form(default="cinematic product showcase, smooth camera motion, professional lighting"),
    num_frames: int = Form(default=81),   # ~3s at 24fps
    steps: int = Form(default=20),
):
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    img = img.resize((832, 480))          # Wan2.1 480P native res

    output = pipe(
        image=img,
        prompt=prompt,
        num_inference_steps=steps,
        num_frames=num_frames,
    )

    out_path = OUT_DIR / f"{uuid.uuid4()}.mp4"
    export_to_video(output.frames[0], str(out_path), fps=24)

    return FileResponse(str(out_path), media_type="video/mp4", filename="output.mp4")
