import os.path

import ebsnrec.models as models
from fuxictr.features import FeatureMap


class ModelFactory(object):
    def __init__(self,data_dir, model_id, **kwargs):
        self.feature_map = FeatureMap.load_from_disk(os.path.join(data_dir, "feature_map.json"))
        self.model_id = model_id
        self.args = kwargs

    def get_model(self):
        model_class = getattr(models, self.model_id, None)
        if model_class is None:
            raise NotImplementedError(f"{model_class}")
        return model_class(feature_map=self.feature_map,
                           model_id=self.model_id,
                           **self.args
                           )