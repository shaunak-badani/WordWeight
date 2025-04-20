from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from schema import Base, GeneratedImage, ExplainedImage

DATABASE_URL = "postgresql://postgres:password@localhost:5432/app_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DataFetcher:

    @classmethod
    def get_explained_images(cls, db):
        explained_images = db.query(ExplainedImage).all()
        return explained_images
    
    @classmethod
    def insert_generated_image(cls, db, *, image_base64, prompt):
        if image_base64 is None or prompt is None:
            raise ValueError("Image or prompt cannot be none!")
        new_image = GeneratedImage(image_base64 = image_base64, prompt = prompt)
        db.add(new_image)
        db.commit()
        return new_image
    
    @classmethod
    def get_image_if_exists(cls, db, *, prompt):
        image = db.query(GeneratedImage).filter_by(prompt=prompt).first()
        # None if no image found, image returned if found
        return image
    
    @classmethod
    def get_explained_image_if_exists(cls, db, *, prompt):
        if prompt is None:
            raise ValueError("Prompt cannot be none!")
        explained_image = db.query(ExplainedImage) \
        .join(ExplainedImage.generated_image) \
        .filter(GeneratedImage.prompt == prompt) \
        .first()
        return explained_image
    

    @classmethod
    def insert_explained_image(cls, db, *, generated_image, masked_image, tokenImportances):
        if generated_image is None or masked_image is None or tokenImportances is None:
            raise ValueError("Image or prompt cannot be none!")
        print(generated_image)
        print(masked_image)
        print(tokenImportances)
        new_explained_image = ExplainedImage(
            generated_image = generated_image, masked_image = masked_image, tokens_imp = tokenImportances)
        db.add(new_explained_image)
        db.commit()
        # return new_explained_image