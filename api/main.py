# api/main.py

import io
import sys
import os
import pickle
import torch
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import CaptionModel
from src.dataset import get_transform
from src.postprocessor import generate_platform_caption
import config


# ── Global variables ──────────────────────────────────────────────────────────
model  = None
vocab  = None
device = None


# ── Lifespan handler ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):

    global model, vocab, device

    print("Loading model...")

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load vocabulary ───────────────────────────────────────────────────────
    vocab_path = os.path.join(config.CHECKPOINT_DIR, "vocab.pkl")
    if not os.path.exists(vocab_path):
        print(f"Vocab not found at {vocab_path}")
    else:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Vocabulary loaded! {len(vocab)} words")

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
    else:
        model = CaptionModel(vocab_size=len(vocab))
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        model.eval()
        model = model.to(device)
        print("Model loaded and ready!")

    yield

    print("Shutting down...")


# ── Initialize FastAPI ────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Image Caption Generator",
    description = "Generate captions for images across different platforms",
    version     = "1.0.0",
    lifespan    = lifespan
)


# ── Helper — preprocess image ─────────────────────────────────────────────────
def preprocess_image(image_bytes):
    """
    Converts raw uploaded image bytes into a normalized tensor.
    Same preprocessing steps used during training.
    """
    # open image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # apply same transforms as training
    transform    = get_transform()
    image_tensor = transform(image)

    # add batch dimension [3,224,224] → [1,3,224,224]
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


# ── Main caption endpoint ─────────────────────────────────────────────────────
@app.post("/caption")
async def generate_caption(
    file    : UploadFile = File(...),
    platform: str        = Form("general")
):
    """
    Upload an image and get a platform specific caption.

    platforms: instagram / linkedin / twitter / email / general
    """

    # check model is loaded
    if model is None or vocab is None:
        return JSONResponse(
            status_code = 503,
            content     = {"error": "Model not loaded yet!"}
        )

    # validate platform
    valid_platforms = ["instagram", "linkedin", "twitter", "email", "general"]
    if platform not in valid_platforms:
        return JSONResponse(
            status_code = 400,
            content     = {"error": f"Invalid platform. Choose from: {valid_platforms}"}
        )

    # convert to token format
    # "instagram" → "<instagram>"
    platform_token = f"<{platform}>"

    try:
        # read uploaded image bytes
        image_bytes = await file.read()

        # preprocess image to tensor
        image_tensor = preprocess_image(image_bytes)
        image_tensor = image_tensor.to(device)

        # generate raw description from our model
        with torch.no_grad():
            raw_caption, attention_maps = model.generate_caption(
                image_tensor,
                vocab,
                platform = platform_token
            )

        # convert raw description → platform specific caption
        # this is where the magic happens!
        # "a woman in white shirt" → "White shirt energy only 🤍 #fashion"
        platform_caption = generate_platform_caption(raw_caption, platform)

        return JSONResponse(content={
            "raw_description" : raw_caption,       # what model sees
            "caption"         : platform_caption,  # polished caption
            "platform"        : platform,
            "words"           : len(platform_caption.split()),
            "status"          : "success"
        })

    except Exception as e:
        return JSONResponse(
            status_code = 500,
            content     = {"error": str(e)}
        )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
async def health_check():
    """
    Check if API is running and model is loaded.
    Visit http://localhost:8000 in browser.
    """
    return {
        "status"      : "running",
        "model_loaded": model is not None,
        "vocab_loaded": vocab is not None
    }


# ── Platforms info ────────────────────────────────────────────────────────────
@app.get("/platforms")
async def get_platforms():
    """
    Returns list of supported platforms and their styles.
    """
    return {
        "platforms": [
            {"name": "instagram", "style": "casual, creative, emoji, hashtags"},
            {"name": "linkedin",  "style": "professional, formal, inspiring"},
            {"name": "twitter",   "style": "short, punchy, fun"},
            {"name": "email",     "style": "descriptive, professional"},
            {"name": "general",   "style": "neutral, standard description"}
        ]
    }


# ── History endpoint (future) ─────────────────────────────────────────────────
@app.get("/history")
async def get_history():
    """
    Returns caption history.
    MongoDB integration coming soon!
    """
    return {
        "message": "History feature coming soon!",
        "status" : "not implemented yet"
    }

# uvicorn api.main:app --reload --port 8000 - use this to run the file
