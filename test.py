import pandas as pd
from ebsnrec.features.feature_encoder import FeatureEncoder

if __name__ == '__main__':

    feature_cols = [{"name":["user_id","event_id","event_group_category","event_group_id"], "active":True, "type":"categorical", "dtype":"str"},
                    {"name":["context_location_similarity"], "active":True, "type":"categorical", "dtype":"float", "encoder":"numeric_bucket", "fill_na":False, "preprocess":"log2","quantile":0.98}

                    ]
    label_col = {"name":"label", "dtype":"float"}

    fe = FeatureEncoder(feature_cols=feature_cols,label_col=label_col,data_dir=".//data")
    df = fe.read_csv(".\\data\\train.csv")
    df = fe.preprocess(df)
    res = fe.fit_transform(df)

    pd.cut(df["context_location_similarity"])


    pass