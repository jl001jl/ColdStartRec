# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import numpy as np
import logging


def evaluate_metrics(y_true, y_pred, metrics, **kwargs):
    result = dict()
    for metric in metrics:
        if metric in ['logloss', 'binary_crossentropy']:
            result[metric] = log_loss(y_true, y_pred, eps=1e-7)
        elif metric == 'AUC':
            result[metric] = roc_auc_score(y_true, y_pred)
        elif metric == "ACC":
            y_pred = np.argmax(y_pred, axis=1)
            result[metric] = accuracy_score(y_true, y_pred)
        else:
            assert "group_index" in kwargs, "group_index is required for GAUC"
            group_index = kwargs["group_index"]
            if metric == "GAUC":
                result[metric] = gAUC(y_true, y_pred, group_index)
            elif metric == "NDCG":
                pass
            elif metric == "MRR":
                result[metric] = mrr(y_true, y_pred, group_index)
            else:
                raise NotImplementedError
    logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in result.items()))
    return result


def gAUC(y_true, y_pred, group_index):
    unique_groups = set(group_index)
    res = 0.0
    for unique_group in unique_groups:
        res += roc_auc_score(y_true[group_index == unique_group], y_pred[group_index == unique_group])
    return res/len(unique_groups)


def mrr(y_true, y_pred, group_index):
    unique_groups = set(group_index)
    res = 0.0
    for unique_group in unique_groups:
        res += group_mrr(y_true[group_index == unique_group], y_pred[group_index == unique_group])
    return res / len(unique_groups)


def group_mrr(y_true, y_pred):
    rank = len(y_true)-(y_true[y_pred.argsort()].argmax())
    return 1/rank