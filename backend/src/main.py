from fastapi import FastAPI, Depends, status, HTTPException
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

# replicate
import replicate

# DB functions
from models import get_db, DataFetcher
from schema import GeneratedImageResponse
from sqlalchemy.orm import Session
from models import create_db

# Image downloading
import requests
import base64

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

@app.get("/traditional")
def query_traditional_model(query: str):
    """
    Query endpoint for the traditional model
    """
    # Pass query to some function
    answer = f"Response to the traditional query : {query}"
    # answer = f(query) 
    return JSONResponse(
        content = {"message": answer}
    )

# @app.get("/users", response_model=list[UserResponse])
# def get_users(db: Session = Depends(get_db)):
#     return DataFetcher.get_users(db)

# @app.get("/users/{user_id}", response_model = UserResponse)
# def get_user(user_id: int, db: Session = Depends(get_db)):
#     user = DataFetcher.get_user(db, user_id)
#     if not user:
#         raise HTTPException(status_code = status.HTTP_404_NOT_FOUND)
#     return user
