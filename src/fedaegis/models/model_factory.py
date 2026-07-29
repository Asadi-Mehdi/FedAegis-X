from .logistic import LogisticModel
from .sgd_classifier import SGDModel


class ModelFactory:

    @staticmethod
    def create(model_name):

        if model_name == "logistic":

            return LogisticModel()

        if model_name == "sgd":

            return SGDModel()

        raise ValueError(

            f"Unknown model : {model_name}"

        )
