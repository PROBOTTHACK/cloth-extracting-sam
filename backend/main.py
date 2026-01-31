from fastapi import FastAPI, UploadFile, Form
import os, uuid, cv2, torch
import numpy as np

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from clothing_regions import get_prompts

# ---------------- CONFIG ----------------
BASE_IMAGES_DIR = "images"
os.makedirs(BASE_IMAGES_DIR, exist_ok=True)

app = FastAPI()

# ---------------- LOAD MODEL (ONCE) ----------------
model = build_sam2(
    "sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2/checkpoints/sam2.1_hiera_tiny.pt",
)
model.eval()
predictor = SAM2ImagePredictor(model)

# ---------------- UTILS ----------------
def save_cutout(image, mask, path):
    h, w, _ = image.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = image
    rgba[..., 3] = mask * 255
    cv2.imwrite(path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

# ---------------- API ----------------
@app.post("/extract")
async def extract(image: UploadFile, type: str = Form(...)):
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(BASE_IMAGES_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Read image
    img_bytes = await image.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Save original
    cv2.imwrite(
        os.path.join(job_dir, "original.png"),
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
    )

    # Get prompts
    h, w, _ = img.shape
    prompt = get_prompts(type, h, w)
    if prompt is None:
        return {"error": "Invalid clothing type"}

    # Run SAM-2
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        predictor.set_image(img)
        masks, scores, _ = predictor.predict(
            point_coords=prompt["points"],
            point_labels=prompt["labels"],
        )

    best_mask = masks[scores.argmax()]
    save_cutout(img, best_mask, os.path.join(job_dir, f"{type}.png"))

    return {
        "job_id": job_id,
        "folder": job_dir,
        "files": os.listdir(job_dir),
    }
