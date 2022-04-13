import io
import json
import logging
import os
from collections import OrderedDict


class FeatureMap(object):
    def __init__(self,):
        self.num_fields = 0
        self.num_features = 0
        self.input_length = 0
        self.feature_specs = OrderedDict()

    def set_feature_index(self):
        logging.info("Set feature index...")
        idx = 0
        for feature, feature_spec in self.feature_specs.items():
            if feature_spec["type"] != "sequence":
                self.feature_specs[feature]["index"] = idx
                idx += 1
            else:
                seq_indexes = [i + idx for i in range(feature_spec["max_len"])]
                self.feature_specs[feature]["index"] = seq_indexes
                idx += feature_spec["max_len"]
        self.input_length = idx

    def get_feature_index(self, feature_type=None):
        feature_indexes = []
        if feature_type is not None:
            if not isinstance(feature_type, list):
                feature_type = [feature_type]
            feature_indexes = [feature_spec["index"] for feature, feature_spec in self.feature_specs.items()
                               if feature_spec["type"] in feature_type]
        return feature_indexes

    @staticmethod
    def load(json_file):
        logging.info("Load feature_map from json: " + json_file)
        with io.open(json_file, "r", encoding="utf-8") as fd:
            feature_map = json.load(fd, object_pairs_hook=OrderedDict)

        f_m_object = FeatureMap()
        f_m_object.num_fields = feature_map["num_fields"]
        f_m_object.num_features = feature_map.get("num_features", None)
        f_m_object.input_length = feature_map.get("input_length", None)
        f_m_object.feature_specs = OrderedDict(feature_map["feature_specs"])

    def save(self, json_file):
        logging.info("Save feature_map to json: " + json_file)
        if not os.path.exists(os.path.dirname(json_file)):
            os.makedirs(os.path.dirname(json_file))
        feature_map = OrderedDict()
        feature_map["num_fields"] = self.num_fields
        feature_map["num_features"] = self.num_features
        feature_map["input_length"] = self.input_length
        feature_map["feature_specs"] = self.feature_specs
        with open(json_file, "w") as fd:
            json.dump(feature_map, fd, indent=4)