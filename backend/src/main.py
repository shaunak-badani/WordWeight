from fastapi import FastAPI, Depends, status, HTTPException
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

# replicate
import replicate

# DB functions
from models import get_db, DataFetcher
from schema import GeneratedImageResponse, ImageUpload, ExplainedImageResponse
from sqlalchemy.orm import Session
from models import create_db

# Image downloading
import requests
import base64
import io
from PIL import Image
from io import BytesIO
import numpy as np

app = FastAPI(root_path='/api')

# list of allowed origins
origins = [
    "http://localhost:5173",
    "http://vcm-45508.vm.duke.edu"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    create_db()


@app.get("/")
async def root():
    return JSONResponse(
        content = {"message": "Hello world!"}
    )

@app.get("/explained-images", response_model = list[ExplainedImageResponse])
def get_explained_images(db: Session = Depends(get_db)):
    return DataFetcher.get_explained_images(db)

@app.get("/generate", response_model=GeneratedImageResponse)
def generate_image(prompt: str, db: Session = Depends(get_db)):
    """
    Query endpoint to generate image given prompt
    """
    prompt = prompt.lower()
    existing_image = DataFetcher.get_image_if_exists(db, prompt = prompt)
    if existing_image is not None:
        return existing_image
    with open("./masked_image.png", "rb") as mask_file:
        image_url = replicate.run(
            "shaunak-badani/wordweight:1cfd38753b1f35fd2edd9a949dcd655dd77b9b2f6840adb8100fd6f3ac298183",
            input={
                "prompt": prompt,
                "mode": "generate",
                "mask_path": mask_file
            }
        )
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_bytes = response.content
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        inserted_image = DataFetcher.insert_generated_image(db, image_base64=image_base64, prompt = prompt)
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e)
        })
    return inserted_image

@app.post("/explain")
def explain_tokens(data: ImageUpload, db: Session = Depends(get_db)):
    """
    Query endpoint for the traditional model
    """
    existing_explained_image = DataFetcher.get_explained_image_if_exists(db, prompt = data.prompt)
    if existing_explained_image:
        return existing_explained_image
    existing_image = DataFetcher.get_image_if_exists(db, prompt = data.prompt)
    try:
        # Decode the base64 image
        header = "data:image/png;base64,"
        if not data.image_base64.startswith(header):
            raise HTTPException(status_code=400, detail="Base 64 not passed!")
        base64_data = data.image_base64[len(header):]
        np_mask = base64_to_mask(base64_data)
        Image.fromarray(np_mask).save("/tmp/maskOutput.png")
        mask = (np_mask > 0).astype(np.uint8)

        current_image = base64_to_mask(existing_image.image_base64)
        dark_overlay = np.zeros_like(current_image)  # black image
        overlay = current_image.copy()
        alpha = 0.4
        overlay[mask == 1] = (
            alpha * dark_overlay[mask == 1] + (1 - alpha) * current_image[mask == 1]
        ).astype(np.uint8)
        # Image.fromarray(overlay).save("./overlaid.png")
        overlayBase64 = mask_to_base64(overlay)
        # tokenImportances = [{ "a" : 2.0 }]

        with open("/tmp/maskOutput.png", "rb") as mask_file:
            tokenImportances = replicate.run(
                "shaunak-badani/wordweight:1cfd38753b1f35fd2edd9a949dcd655dd77b9b2f6840adb8100fd6f3ac298183",
                input={
                    "prompt": data.prompt,
                    "mode": "explain",
                    "mask_path": mask_file
                }
            )
        DataFetcher.insert_explained_image(db, generated_image=existing_image, 
                    masked_image = overlayBase64, tokenImportances=tokenImportances)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(
        content = {"message": tokenImportances}
    )


def base64_to_mask(base64_str: str) -> np.ndarray:
    decoded = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(decoded)).convert("RGB")
    mask = np.array(img).astype(np.uint8)
    return mask

def mask_to_base64(mask: np.ndarray) -> str:
    img = Image.fromarray((mask * 255).astype(np.uint8))  # assuming binary mask
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")