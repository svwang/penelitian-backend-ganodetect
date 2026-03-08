from sqlalchemy import Column, Integer, String, Float, ForeignKey, TIMESTAMP
from sqlalchemy.sql import func
from .connectionDb import Base

class Inference(Base):
    __tablename__ = "inferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), default=1)

    label = Column(String(50))
    confidence = Column(Float)
    inference_time = Column(Float)

    gradcam_image = Column(String(255))

    created_at = Column(TIMESTAMP, server_default=func.now())