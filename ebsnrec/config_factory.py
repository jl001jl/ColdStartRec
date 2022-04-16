import json
import logging
import os.path
from collections import OrderedDict

from ebsnrec.util import get_dict_from_config_path, update_dict_by_dict_recursively


class ConfigFactory(object):
    def __init__(self, config_dir: str = "D:\EBSNRec\ColdStartRec\config"):
        self.config_dir = config_dir

    def get_config(self, model="WideDeep", dataset="meetup_sg", **kwargs):
        config_dict = dict()
        self.__build_common_config(config_dict, model=model, dataset=dataset, **kwargs)
        self.__build_creator_config(config_dict, model=model, dataset=dataset, **kwargs)
        self.__build_feature_config(config_dict, model=model, dataset=dataset, **kwargs)
        self.__build_model_config(config_dict, model=model, dataset=dataset, **kwargs)
        self.__build_evaluation_config(config_dict, model=model, dataset=dataset, **kwargs)
        self.__build_log_config(config_dict, model=model, dataset=dataset, **kwargs)
        return config_dict

    def __build_common_config(self, config_dict, model="WideDeep", dataset="meetup_sg", **kwargs):
        config_dict["common_config"] = get_dict_from_config_path(
            os.path.join(self.config_dir, "base_config", "common.yaml"))

    def __build_creator_config(self, config_dict, model="WideDeep", dataset="meetup_sg", **kwargs):
        config = self.__load_base_config(config_type="creator", key="base")
        dataset_spec_config = self.__load_base_config(config_type="creator", key=dataset)
        update_dict_by_dict_recursively(config, dataset_spec_config)
        config_dict["creator_config"] = config

    def __build_feature_config(self, config_dict, model="WideDeep", dataset="meetup_sg", **kwargs):
        config = self.__load_base_config(config_type="feature", key="base")
        model_spec_config = self.__load_base_config(config_type="feature", key=model)
        update_dict_by_dict_recursively(config, model_spec_config)
        config_dict["feature_config"] = config

    def __build_model_config(self, config_dict, model="WideDeep", dataset="meetup_sg", **kwargs):
        config = self.__load_base_config(config_type="model", key="base")
        model_spec_config = self.__load_base_config(config_type="model", key=model)
        update_dict_by_dict_recursively(config, model_spec_config)
        config_dict["model_config"] = config

    def __build_evaluation_config(self, config_dict, model="WideDeep", dataset="meetup_sg", **kwargs):
        config = self.__load_base_config(config_type="evaluation", key="base")
        config_dict["evaluation_config"] = config

    def __build_log_config(self, config_dict, model="WideDeep", dataset="meetup_sg", **kwargs):
        config = {
            "level": logging.INFO,
            "format": '%(asctime)s %(levelname)s %(message)s'
        }
        config_dict["log_config"] = config

    def __load_base_config(self, config_type="creator", key="base"):
        path = os.path.join(self.config_dir, "base_config", config_type + '.yaml')
        config_dict = get_dict_from_config_path(path)
        return config_dict.get(key, dict())


if __name__ == '__main__':
    config_factor = ConfigFactory()
    res = config_factor.get_config(dataset='meetup_ny')
    print(json.dumps(OrderedDict(res), indent=4))
