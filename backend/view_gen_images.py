from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, GeneratedImage  # Replace with your actual model import

# Replace with your actual database URL
DATABASE_URL = "postgresql://postgres:password@localhost:5432/app_db"

# Set up engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def view_generated_images():
    session = SessionLocal()
    try:
        images = session.query(GeneratedImage).all()
        for img in images:
            print(f"ID: {img.id}")
            print(f"Prompt: {img.prompt}")
            print(f"Image (base64): {img.image_base64[:100]}...")  # Print first 100 chars
            print(f"Created at: {img.created_at}")
            print("-" * 40)
    finally:
        session.close()

if __name__ == "__main__":
    view_generated_images()