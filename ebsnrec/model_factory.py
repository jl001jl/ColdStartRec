import os.path

from fuxictr.features import FeatureMap
from fuxictr.pytorch.models import DeepFM, DNN





class ModelFactory(object):
    def __init__(self,data_dir,model_root, model_id, metrics,verbose,optimizer,loss,learning_rate,**kwargs):
        self.feature_map = FeatureMap.load_from_disk(os.path.join(data_dir, "feature_map.json"))
        self.model_id = model_id
        self.model_root = model_root
        self.metrics = metrics
        self.verbose = verbose
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate


    def get_model(self):
        if self.model_id == "DNN":
            model_class = DNN
        else:
            model_class = DeepFM

        return model_class(feature_map=self.feature_map,
                           model_root=self.model_root,
                           metrics=self.metrics,
                           verbose=self.verbose,
                           optimizer=self.optimizer,
                           loss=self.loss,
                           learning_rate=self.learning_rate)