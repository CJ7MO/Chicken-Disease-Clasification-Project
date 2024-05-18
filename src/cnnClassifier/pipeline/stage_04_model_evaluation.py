from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation import ModelEvaluation
from cnnClassifier import logger


STAGE_NAME = "Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass


    def main(self):
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        model_evaluation = ModelEvaluation(val_config)
        model_evaluation.model_evaluation()
        model_evaluation.save_score()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e