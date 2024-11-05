from pathlib import Path
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf

from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till:
            for layer in model.layers[:freeze_till]:
                layer.trainable = False
        
        x = tf.keras.layers.Flatten()(model.output)
        output = tf.keras.layers.Dense(classes, activation='softmax')(x)
        model = tf.keras.Model(model.input, output)
        
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), 
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()
        return model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)