from fastapi import FastAPI, UploadFile, Form
import os, uuid, cv2, torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from clothing_regions import get_prompts

BASE_DIR = "images"
os.makedirs(BASE_DIR, exist_ok=True)

app = FastAPI()

# ---- Load model ONCE ----
model = build_sam2(
    "sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2/checkpoints/sam2.1_hiera_tiny.pt"
)
model.eval()
predictor = SAM2ImagePredictor(model)

def save_cutout(img, mask, path):
    rgba = np.zeros((*img.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = img
    rgba[..., 3] = mask * 255
    cv2.imwrite(path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

@app.post("/extract")
async def extract(image: UploadFile, type: str = Form(...)):
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(BASE_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    img_bytes = await image.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(f"{job_dir}/original.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    h, w, _ = img.shape
    prompt = get_prompts(type, h, w)

    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        predictor.set_image(img)
        masks, scores, _ = predictor.predict(
            point_coords=prompt["points"],
            point_labels=prompt["labels"]
        )

    mask = masks[scores.argmax()]
    save_cutout(img, mask, f"{job_dir}/{type}.png")

    return {
        "job_id": job_id,
        "folder": job_dir
    }
