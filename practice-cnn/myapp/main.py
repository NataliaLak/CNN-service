import torch
from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from myapp.model import Model
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# load model
model = Model()

# app
app = FastAPI(title='Symbol detection', docs_url='/docs')

# api
@app.post('/api/predict')
def predict(image: str = Body(..., description='image pixels list')):
    # Определяем трансформации
    transform = Compose([ToTensor(), Normalize([0.5], [0.5])])

    # Преобразуем строку в список пикселей
    pixels = list(map(int, image[1:-1].split(',')))

    # Преобразуем список пикселей в изображение PIL
    img = Image.fromarray(np.array(pixels).reshape(28, 28).astype(np.uint8))

    # Применяем трансформации
    img_tensor = transform(img)  # Преобразуем (28, 28) в (1, 28, 28)
    img_tensor = img_tensor.unsqueeze(0)  # Добавляем размерность батча: (1, 1, 28, 28)
    
    # Отладка: печать размера тензора
    logger.debug(f"Image tensor size: {img_tensor.size()}")

    # Предсказание
    pred = model.predict(img_tensor)
    
    return {'prediction': pred}