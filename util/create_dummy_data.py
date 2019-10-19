import os
import numpy as np
from PIL import Image

height = 128
width = 128
channel = 3
num = 20
save_dir = 'data'

for i in range(num):
  a = np.random.randint(0, 255, (128,128,3))
  pilImg = Image.fromarray(np.uint8(a))
  pilImg.save(os.path.join(save_dir, str(i) + '.png'))