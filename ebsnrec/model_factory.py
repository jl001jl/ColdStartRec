import os.path

from ebsnrec.models import DIERM
from fuxictr.features import FeatureMap
from fuxictr.pytorch.models import DeepFM, DNN, WideDeep, DIN


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
        if self.model_id == "WideDeep":
            model_class = WideDeep
        elif self.model_id == "DeepFM":
            model_class = DeepFM
        elif self.model_id == "DIN":
            model_class = DIN
        elif self.model_id == "DIERM":
            model_class = DIERM
        elif self.model_id == "MostPopular":
            model_class = DNN
        elif self.model_id == "ContextRec":
            model_class = DNN
        elif self.model_id == "EventRec":
            model_class = DNN
        elif self.model_id == "JointRec":
            model_class = DNN
        else:
            model_class = DNN
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