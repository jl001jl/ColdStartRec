from ebsnrec.config_factory import ConfigFactory
from ebsnrec.train_eval import evaluation_once


if __name__ == '__main__':
    config =  ConfigFactory().get_config(dataset="meetup_ny",model="DNN")
    evaluation_once(**config)
