import gc
import logging
import os

from ebsnrec.util import get_signature, create_signature, save_signature
from fuxictr.datasets import save_hdf5
from fuxictr.features import FeatureEncoder


class DatasetBuilder(object):
    def __init__(self, data_dir: str, save_dir: str, feature_cols=None, spec_cols=None, **kwargs):
        self.save_dir = save_dir
        self.data_dir = data_dir
        self.feature_encoder = FeatureEncoder(feature_cols=feature_cols,
                                              spec_cols=spec_cols,
                                              data_root=save_dir)
        self._signature = create_signature(get_signature(data_dir), feature_cols, spec_cols)

    def build_data(self):
        if self.check_can_skip():
            logging.warning(f"skip build data {self.save_dir}")
            return
        self._build_data()
        save_signature(self._signature,self.save_dir)

    def get_feature_map(self):
        return self.feature_encoder.feature_map

    def _build_data(self):
        feature_encoder = self.feature_encoder
        train_data = os.path.join(self.data_dir, "train.csv")
        train_ddf = feature_encoder.read_csv(train_data)

        # fit and transform train_ddf
        train_ddf = feature_encoder.preprocess(train_ddf)
        train_array = feature_encoder.fit_transform(train_ddf)

        save_hdf5(train_array, os.path.join(feature_encoder.data_dir, 'train.h5'))
        del train_array, train_ddf
        gc.collect()

        valid_data = os.path.join(self.data_dir, "valid.csv")
        valid_ddf = feature_encoder.read_csv(valid_data)
        valid_ddf = feature_encoder.preprocess(valid_ddf)
        valid_array = feature_encoder.transform(valid_ddf)

        save_hdf5(valid_array, os.path.join(feature_encoder.data_dir, 'valid.h5'))
        del valid_array, valid_ddf
        gc.collect()

        # Transfrom test_ddf
        test_data = os.path.join(self.data_dir, "test.csv")
        test_ddf = feature_encoder.read_csv(test_data)

        test_ddf = feature_encoder.preprocess(test_ddf)
        test_array = feature_encoder.transform(test_ddf)

        save_hdf5(test_array, os.path.join(feature_encoder.data_dir, 'test.h5'))

    def check_can_skip(self):
        last_signature = get_signature(self.save_dir)
        if last_signature is not None and last_signature == self._signature:
            return True
        else:
            return False
