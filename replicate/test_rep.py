from cog import BasePredictor, BaseModel, File, Path
from typing import Union, List, Any
import io

class Output(BaseModel):
    token: str
    importance: float

class Predictor(BasePredictor):
    def predict(self) -> Any:
        return [
            Output(token="hello", importance = 2.421),
            Output(token="hello", importance = 2.421)
        ]