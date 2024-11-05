from cnnClassifier.components.model_trainer import Training
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger


STAGE_NAME = 'Model trainer'

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        trainer = Training(training_config)
        trainer.get_base_model()
        trainer.make_train_valid_generators()
        trainer.train()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.run()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e