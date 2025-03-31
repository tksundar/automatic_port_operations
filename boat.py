import numpy as np
import os

# Collect images in differnet lists based on the confidence score.We will create 10 bins

test_dir = '/kaggle/input/boats-zip-file/TEST_BOATS/all'


def get_images_in_confidence_range(model):
    '''

    '''


images = os.listdir(test_dir)
image_label_map = {}

for i, image in enumerate(images):
    '''
    '''
    path = test_dir + '/' + image
    img = tf.keras.utils.load_img(path, target_size=(height, width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array, verbose=False)
    score = tf.nn.softmax(predictions[0])
    max_score = np.max(score)
    image_label_map[path] = class_names[np.argmax(score)]

return image_label_map