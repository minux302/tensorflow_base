import os
import cv2
import numpy as np
from PIL import Image
import random


class Loader:
  def __init__(self,
               data_dir,
               batch_size=10,
               input_height=128,
               input_width=128):

    self.batch_size = batch_size
    self.input_height = input_height
    self.input_width = input_width

    img_list = os.listdir(data_dir)
    self.image_paths = [os.path.join(data_dir, img_name) for img_name in img_list]

    self.indexes = np.arange(len(img_list), dtype=int)

  def __getitem__(self, batch_id):
    from_b = batch_id * self.batch_size
    to_b = (batch_id + 1) * self.batch_size
    ba_ids = self.indexes[from_b:to_b]
    inputs, labels = self._get_data_from_indexes(ba_ids)

    return inputs, labels

  def __len__(self):
      return int(len(self.image_paths) / self.batch_size)

  # for shuffle indexes
  def on_epoch_end(self):
    indexes = np.arange(len(self.image_paths), dtype=int)
    np.random.shuffle(indexes)
    self.indexes = indexes

  def _get_data_from_indexes(self, ids):
      images = np.array([self._preprocess_image(self.image_paths[i]) for i in ids])
      labels = np.array([np.random.randint(10) for _ in ids])  # dummy
      return images, labels

  def _preprocess_image(self, image_path):
      # load and reshape
      image = np.array(Image.open(image_path).convert("RGB").resize((self.input_height, self.input_width)))

      return image

  

if __name__ == "__main__":
  data_dir = "data"
  input_height = 128
  input_width = 128
  batch_size = 3
  loader = Loader(data_dir=data_dir,
                  input_height=input_height,
                  input_width=input_width,
                  batch_size=batch_size)

  for i in range(len(loader)):
    image_batch, label_batch = loader[i]
    print(image_batch.shape)
    print(label_batch.shape)