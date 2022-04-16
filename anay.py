import pandas as pd

df = pd.read_csv("./data/meetup_ny/dataset/test.csv")
sb = df[["label","user_join_event_group","context_social_similarity"]]
print('1')