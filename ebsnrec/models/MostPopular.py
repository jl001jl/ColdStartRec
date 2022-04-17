import torch

from fuxictr.pytorch.layers import LR_Layer
from fuxictr.pytorch.models import BaseModel


class MostPopular(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="MostPopular",
                 gpu=-1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 regularizer=None,
                 **kwargs):
        super(MostPopular, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       embedding_regularizer=regularizer,
                                       net_regularizer=regularizer,
                                       **kwargs)
        self.lr_layer = LR_Layer(feature_map,
                                 output_activation=self.get_output_activation(task),
                                 use_bias=True)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()
        self._stop_training = True


    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        X = X.to(dtype=torch.double)
        y = y.to(dtype=torch.double)
        return_dict = {"y_pred": X, "y_true": y}
        return return_dict