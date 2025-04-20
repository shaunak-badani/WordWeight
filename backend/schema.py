from sqlalchemy import Column, Integer, String, Text, DateTime, func, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import JSONB
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any

Base = declarative_base()

class GeneratedImage(Base):
    __tablename__ = "generated_images"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    image_base64 = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    explanations = relationship("ExplainedImage", back_populates="generated_image")

    class Config:
        orm_mode = True

    def __repr__(self):
        return f"<GenImage id={self.id} name={self.prompt} created_at={self.created_at}>"

class ExplainedImage(Base):
    __tablename__ = "explained_images"

    id = Column(Integer, primary_key=True, index=True)
    generated_image_id = Column(Integer, ForeignKey("generated_images.id"), nullable=False)
    tokens_imp = Column(JSONB, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    masked_image = Column(JSONB, nullable=False)

    # Relationship to fetch the associated generated image
    generated_image = relationship("GeneratedImage", back_populates="explanations") 

class GeneratedImageResponse(BaseModel):
    id: int
    prompt: str
    image_base64: str
    created_at: datetime

    class Config:
        orm_mode = True  # This tells Pydantic to treat the SQLAlchemy model as a dict


class ImageUpload(BaseModel):
    prompt: str
    image_base64: str

class TokenImportance(BaseModel):
    word: str
    importance: float

class ExplainedImageResponse(BaseModel):
    id: int
    generated_image_id: int
    tokens_imp: list[TokenImportance]  # Adjust type depending on your tokens_imp structure
    created_at: datetime
    masked_image: str
    generated_image: GeneratedImageResponse

    class Config:
        orm_mode = True
        from_attributes=True