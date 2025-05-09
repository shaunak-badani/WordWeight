from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from schema import Base, GeneratedImage, ExplainedImage, ExplainedImageResponse # Replace with your actual model import

import base64
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

# Replace with your actual database URL
DATABASE_URL = "postgresql://postgres:password@localhost:5432/app_db"

# Set up engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def delete_explained_image_by_id(image_id: int):
    session = SessionLocal()
    try:
        image = session.query(ExplainedImage).filter(ExplainedImage.id == image_id).first()
        if image:
            session.delete(image)
            session.commit()
            print(f"Deleted image with ID {image_id}")
        else:
            print(f"No image found with ID {image_id}")
    finally:
        session.close()

def decode_and_show_base64_image(base64_str: str, title="Masked Image"):
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    plt.imshow(image)
    plt.axis('off')
    plt.title(title)
    plt.show()

def show_explained_image_by_id(image_id: int):
    session = SessionLocal()
    try:
        img = session.query(ExplainedImage).filter(ExplainedImage.id == image_id).first()
        if img:
            decode_and_show_base64_image(img.masked_image, title=f"Masked Image ID: {img.id}")
        else:
            print(f"No explained image found with ID {image_id}")
    finally:
        session.close()

# Delete image with ID = 2
# delete_image_by_id(1)
# delete_image_by_id(2)
# delete_image_by_id(6)
# delete_image_by_id(7)
# delete_explained_image_by_id(1)

def view_generated_images():
    session = SessionLocal()
    try:
        images = session.query(GeneratedImage).all()
        for img in images:
            print(f"ID: {img.id}")
            print(f"Prompt: {img.prompt}")
            print(f"Image (base64): {img.image_base64[:200]}...")  # Print first 100 chars
            print(f"Created at: {img.created_at}")
            print("-" * 40)
    finally:
        session.close()

def view_explained_images():
    session = SessionLocal()
    try:
        images = session.query(ExplainedImage).all()
        for img in images:
            print(f"ID: {img.id}")
            print(f"Prompt: {img.generated_image.prompt}")
            print(f"Masked Image (base64): {img.masked_image[:200]}...")  # Print first 100 chars
            print(f"Created at: {img.generated_image.created_at}")
            print(f"Token imp: {img.tokens_imp}...")  # Print first 100 chars
            print("-" * 40)
    finally:
        session.close()

if __name__ == "__main__":
    # view_generated_images()
    delete_explained_image_by_id(8)
    # view_explained_images()
    # show_explained_image_by_id(4)