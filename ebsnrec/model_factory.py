import os.path

import ebsnrec.models as models
from fuxictr.features import FeatureMap


class ModelFactory(object):
    def __init__(self,data_dir,model_root, model_id, metrics,verbose,optimizer,loss,learning_rate,spec_cols,gpu,**kwargs):
        self.feature_map = FeatureMap.load_from_disk(os.path.join(data_dir, "feature_map.json"))
        self.model_id = model_id
        self.model_root = model_root
        self.metrics = metrics
        self.verbose = verbose
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate
        self.spec_cols = spec_cols
        self.gpu = gpu


    def get_model(self):
        model_class = getattr(models, self.model_id, None)
        if model_class is None:
            raise NotImplementedError(f"{model_class}")
        return model_class(feature_map=self.feature_map,
                           model_root=self.model_root,
                           metrics=self.metrics,
                           verbose=self.verbose,
                           optimizer=self.optimizer,
                           loss=self.loss,
                           learning_rate=self.learning_rate,
                           spec_cols = self.spec_cols,
                           gpu=self.gpu
                           )