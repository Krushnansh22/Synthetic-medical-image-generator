import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def display_metadata(data, index):
    image_metadata = pd.DataFrame(data.iloc[index,:]).reset_index()
    image_metadata.columns = ['Feature', 'Value']
    print('\nMetadata for displayed image:')
    print(image_metadata)


def display_image(data):
    index = np.random.randint(0,data.shape[0])
    image_id = data['image_id'][index]
    if not image_id.endswith('.jpg'):
        image_id += '.jpg'
    file1 = os.path.join('data', 'HAM10000_images_part_1', image_id)
    file2 = os.path.join('data', 'HAM10000_images_part_2', image_id)
    
    if not os.path.exists(file1) and not os.path.exists(file2):
        raise FileNotFoundError
    elif os.path.exists(file1):
        image = Image.open(file1)
    else:
        image = Image.open(file2)
    
    display_metadata(data, index)
    
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    