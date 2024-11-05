import numpy as np
import tensorflow as tf
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
    
    def predict(self):
        # model = tf.keras.models.load_model(os.path.join("artifacts","training", "model.h5"))
        model = tf.keras.models.load_model(os.path.join("model", "model.h5"))

        image_name = self.filename
        input_image = tf.keras.preprocessing.image.load_img(image_name, target_size=(224, 224))
        input_image = tf.keras.preprocessing.image.img_to_array(input_image)
        input_image = np.expand_dims(input_image, axis=0)

        result = np.argmax(model.predict(input_image), axis=1)

        if result[0] == 1:
            prediction = 'Normal'
        else:
            prediction = 'Adenocarcinoma Cancer'
        
        return [{"image": prediction}]
