from ebsnrec.config_factory import ConfigFactory
from ebsnrec.train_eval import evaluation_once


if __name__ == '__main__':
    config = ConfigFactory("./config").get_config(dataset="meetup_sg",model="DIN")
    evaluation_once(**config)
