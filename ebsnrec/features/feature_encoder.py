import logging
import os
import pickle
from collections import Counter

import numpy as np
import sklearn.preprocessing as sklearn_preprocess
import pandas as pd

from ebsnrec.features.feature_map import FeatureMap
from ebsnrec.features.preprocess import Normalizer, Tokenizer


class NumericBucketEncoder:
    def __init__(self,num_buckets=10, max_quantile=1.0):
        self.num_buckets = num_buckets
        self.max_quantile = max_quantile
        pass

    def fit(self, values):
        pass

    def transform(self):
        pass



class FeatureEncoder(object):
    def __init__(self,
                 feature_cols=[],
                 label_col={},
                 data_dir="../data/",
                 **kwargs):
        logging.info("Set up feature encoder...")
        self.data_dir =  data_dir
        self.pickle_file = os.path.join(self.data_dir, "feature_encoder.pkl")
        self.json_file = os.path.join(self.data_dir, "feature_map.json")
        self.feature_cols = self._complete_feature_cols(feature_cols)
        self.label_col = label_col
        self.feature_map = FeatureMap()
        self.encoders = dict()

    def _complete_feature_cols(self, feature_cols):
        full_feature_cols = []
        for col in feature_cols:
            name_or_namelist = col["name"]
            if isinstance(name_or_namelist, list):
                for _name in name_or_namelist:
                    _col = col.copy()
                    _col["name"] = _name
                    full_feature_cols.append(_col)
            else:
                full_feature_cols.append(col)
        return full_feature_cols

    def read_csv(self, data_path):
        logging.info("Reading file: " + data_path)
        all_cols = self.feature_cols + [self.label_col]
        dtype_dict = dict((x["name"], eval(x["dtype"]) if isinstance(x["dtype"], str) else x["dtype"])
                          for x in all_cols)
        ddf = pd.read_csv(data_path, dtype=dtype_dict, memory_map=True)
        return ddf

    def preprocess(self, ddf, fill_na=True):
        logging.info("Preprocess feature columns...")
        all_cols = [self.label_col] + self.feature_cols[::-1]
        for col in all_cols:
            name = col["name"]
            fill_na = col.get("fill_na", True)
            if fill_na and name in ddf.columns and ddf[name].isnull().values.any() and fill_na:
                ddf[name] = self._fill_na(col, ddf[name])
            if "preprocess" in col and col["preprocess"] != "":
                preprocess_fn = getattr(self, col["preprocess"])
                ddf[name] = preprocess_fn(ddf, name)
        active_cols = [self.label_col["name"]] + [col["name"] for col in self.feature_cols if col["active"]]
        ddf = ddf.loc[:, active_cols]
        return ddf

    def _fill_na(self, col, series):
        na_value = col.get("na_value")
        if na_value is not None:
            return series.fillna(na_value)
        elif col["dtype"] in ["str", str]:
            return series.fillna("")
        else:
            raise RuntimeError("Feature column={} requires to assign na_value!".format(col["name"]))

    def fit_transform(self, ddf, min_categr_count=1, num_buckets=10, **kwargs):
        self.fit(ddf, min_categr_count=min_categr_count, num_buckets=num_buckets, **kwargs)
        data_array = self.transform(ddf)
        return data_array

    def fit(self, ddf, min_categr_count=1, num_buckets=10, **kwargs):
        logging.info("Fit feature encoder...")
        self.feature_map.num_fields = 0
        for col in self.feature_cols:
            if col["active"]:
                logging.info("Processing column: {}".format(col))
                name = col["name"]
                self.fit_feature_col(col, ddf[name].values,
                                     min_categr_count=min_categr_count,
                                     num_buckets=num_buckets)
                self.feature_map.num_fields += 1
        self.feature_map.set_feature_index()
        self.save_pickle(self.pickle_file)
        self.feature_map.save(self.json_file)
        logging.info("Set feature encoder done.")

    def fit_feature_col(self, feature_column, feature_values, min_categr_count=1, num_buckets=10):
        name = feature_column["name"]
        feature_type = feature_column["type"]
        feature_source = feature_column.get("source", "")
        self.feature_map.feature_specs[name] = {"source": feature_source,
                                                "type": feature_type}
        if "min_categr_count" in feature_column:
            min_categr_count = feature_column["min_categr_count"]
            self.feature_map.feature_specs[name]["min_categr_count"] = min_categr_count
        if "embedding_dim" in feature_column:
            self.feature_map.feature_specs[name]["embedding_dim"] = feature_column["embedding_dim"]
        if feature_type == "numeric":
            normalizer_name = feature_column.get("normalizer", None)
            if normalizer_name is not None:
                normalizer = Normalizer(normalizer_name)
                normalizer.fit(feature_values)
                self.encoders[name + "_normalizer"] = normalizer
            self.feature_map.num_features += 1
        elif feature_type == "categorical":
            encoder = feature_column.get("encoder", "")
            if encoder != "":
                self.feature_map.feature_specs[name]["encoder"] = encoder
            if encoder == "":
                tokenizer = Tokenizer(min_freq=min_categr_count,
                                      na_value=feature_column.get("na_value", ""))
                if "share_embedding" in feature_column:
                    self.feature_map.feature_specs[name]["share_embedding"] = feature_column["share_embedding"]
                    tokenizer.set_vocab(self.encoders["{}_tokenizer".format(feature_column["share_embedding"])].vocab)
                else:
                    if self.is_share_embedding_with_sequence(name):
                        tokenizer.fit_on_texts(feature_values, use_padding=True)
                    else:
                        tokenizer.fit_on_texts(feature_values, use_padding=False)
                if "pretrained_emb" in feature_column:
                    logging.info("Loading pretrained embedding: " + name)
                    self.feature_map.feature_specs[name]["pretrained_emb"] = "pretrained_{}.h5".format(name)
                    self.feature_map.feature_specs[name]["freeze_emb"] = feature_column.get("freeze_emb", True)
                    tokenizer.load_pretrained_embedding(name,
                                                        feature_column["pretrained_emb"],
                                                        feature_column["embedding_dim"],
                                                        os.path.join(self.data_dir, "pretrained_{}.h5".format(name)),
                                                        feature_dtype=feature_column.get("dtype"),
                                                        freeze_emb=feature_column.get("freeze_emb", True))
                if tokenizer.use_padding:  # update to account pretrained keys
                    self.feature_map.feature_specs[name]["padding_idx"] = tokenizer.vocab_size - 1
                self.encoders[name + "_tokenizer"] = tokenizer
                self.feature_map.num_features += tokenizer.vocab_size
                self.feature_map.feature_specs[name]["vocab_size"] = tokenizer.vocab_size
            elif encoder == "numeric_bucket":
                num_buckets = feature_column.get("num_buckets", num_buckets)
                max_quantile = feature_column.get("quantile",1.0)
                numeric_bucket_encoder = NumericBucketEncoder(num_buckets=num_buckets, max_quantile=max_quantile)
                numeric_bucket_encoder.fit(feature_values)
                self.feature_map.feature_specs[name]["vocab_size"] = num_buckets
                self.feature_map.num_features += num_buckets
                self.encoders[name + "_encoder"] = numeric_bucket_encoder
            elif encoder == "hash_bucket":
                num_buckets = feature_column.get("num_buckets", num_buckets)
                uniques = Counter(feature_values)
                num_buckets = min(num_buckets, len(uniques))
                self.feature_map.feature_specs[name]["vocab_size"] = num_buckets
                self.feature_map.num_features += num_buckets
                self.encoders[name + "_num_buckets"] = num_buckets
        elif feature_type == "sequence":
            encoder = feature_column.get("encoder", "MaskedAveragePooling")
            splitter = feature_column.get("splitter", " ")
            na_value = feature_column.get("na_value", "")
            max_len = feature_column.get("max_len", 0)
            padding = feature_column.get("padding", "post")
            tokenizer = Tokenizer(min_freq=min_categr_count, splitter=splitter,
                                  na_value=na_value, max_len=max_len, padding=padding)
            if "share_embedding" in feature_column:
                if feature_column.get("max_len") is None:
                    tokenizer.fit_on_texts(feature_values, use_padding=True)  # Have to get max_len even share_embedding
                self.feature_map.feature_specs[name]["share_embedding"] = feature_column["share_embedding"]
                tokenizer.set_vocab(self.encoders["{}_tokenizer".format(feature_column["share_embedding"])].vocab)
            else:
                tokenizer.fit_on_texts(feature_values, use_padding=True)
            if "pretrained_emb" in feature_column:
                logging.info("Loading pretrained embedding: " + name)
                self.feature_map.feature_specs[name]["pretrained_emb"] = "pretrained_{}.h5".format(name)
                self.feature_map.feature_specs[name]["freeze_emb"] = feature_column.get("freeze_emb", True)
                tokenizer.load_pretrained_embedding(name,
                                                    feature_column["pretrained_emb"],
                                                    feature_column["embedding_dim"],
                                                    os.path.join(self.data_dir, "pretrained_{}.h5".format(name)),
                                                    feature_dtype=feature_column.get("dtype"),
                                                    freeze_emb=feature_column.get("freeze_emb", True))
            self.encoders[name + "_tokenizer"] = tokenizer
            self.feature_map.num_features += tokenizer.vocab_size
            self.feature_map.feature_specs[name].update({"encoder": encoder,
                                                         "padding_idx": tokenizer.vocab_size - 1,
                                                         "vocab_size": tokenizer.vocab_size,
                                                         "max_len": tokenizer.max_len})
        else:
            raise NotImplementedError("feature_col={}".format(feature_column))

    def transform(self, ddf):
        logging.info("Transform feature columns...")
        data_arrays = []
        for feature, feature_spec in self.feature_map.feature_specs.items():
            feature_type = feature_spec["type"]
            if feature_type == "numeric":
                numeric_array = ddf.loc[:, feature].fillna(0).apply(lambda x: float(x)).values
                normalizer = self.encoders.get(feature + "_normalizer")
                if normalizer:
                    numeric_array = normalizer.normalize(numeric_array)
                data_arrays.append(numeric_array)
            elif feature_type == "categorical":
                encoder = feature_spec.get("encoder", "")
                if encoder == "":
                    data_arrays.append(self.encoders.get(feature + "_tokenizer") \
                                       .encode_category(ddf.loc[:, feature].values))
                elif encoder == "numeric_bucket":
                    raise NotImplementedError
                elif encoder == "hash_bucket":
                    raise NotImplementedError
            elif feature_type == "sequence":
                data_arrays.append(self.encoders.get(feature + "_tokenizer") \
                                   .encode_sequence(ddf.loc[:, feature].values))
        label_name = self.label_col["name"]
        if ddf[label_name].dtype != np.float64:
            ddf.loc[:, label_name] = ddf.loc[:, label_name].apply(lambda x: float(x))
        data_arrays.append(ddf.loc[:, label_name].values)  # add the label column at last
        data_arrays = [item.reshape(-1, 1) if item.ndim == 1 else item for item in data_arrays]
        data_array = np.hstack(data_arrays)
        return data_array

    def is_share_embedding_with_sequence(self, feature):
        for col in self.feature_cols:
            if col.get("share_embedding", None) == feature and col["type"] == "sequence":
                return True
        return False

    def load_pickle(self, pickle_file=None):
        """ Load feature encoder from cache """
        if pickle_file is None:
            pickle_file = self.pickle_file
        logging.info("Load feature_encoder from pickle: " + pickle_file)
        if os.path.exists(pickle_file):
            pickled_feature_encoder = pickle.load(open(pickle_file, "rb"))

            return pickled_feature_encoder
        raise IOError("pickle_file={} not valid.".format(pickle_file))

    def save_pickle(self, pickle_file):
        logging.info("Pickle feature_encoder: " + pickle_file)
        if not os.path.exists(os.path.dirname(pickle_file)):
            os.makedirs(os.path.dirname(pickle_file))
        pickle.dump(self, open(pickle_file, "wb"))

    def load_json(self, json_file):
        self.feature_map.load(json_file)

    def log2(self, ddf , col ):
        return np.log2(ddf[col]+1)
