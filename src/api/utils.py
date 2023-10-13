from PIL import Image
from io import BytesIO
import numpy as np

def read_imagefile(file) -> np.ndarray:
    image = np.array(Image.open(BytesIO(file)))
    return image
