import copy
import json

import textdistance
from tqdm.auto import tqdm

TRAINING_TRACKS = "./data/data/train-tracks_2.json"

with open(TRAINING_TRACKS, "r") as f_r:
    train_tracks = json.load(f_r)
    track_ids = list(train_tracks.keys())


train_queries = []
for track_id in track_ids:
    train_queries.extend(train_tracks[track_id]["nl"])

for track_id in tqdm(track_ids, desc="Matching negative description"):
    nls = train_tracks[track_id]["nl"]
    negs = []
    for nl in nls:
        for neg in list(set(train_queries)):
            similarity = textdistance.overlap.normalized_similarity(nl.split(), neg.split())
            negs.append((neg, similarity))

    negs = [k[0] for k in sorted(negs, key=lambda tup: tup[1], reverse=True)]
    negs = negs[50:]
    train_tracks[track_id]["neg_nl"] = negs

my_train = dict()
my_validation = dict()

for i, track_id in enumerate(track_ids):
    if "S01" in train_tracks[track_id]["frames"][0]:
        my_validation[track_id] = copy.deepcopy(train_tracks[track_id])
    else:
        my_train[track_id] = copy.deepcopy(train_tracks[track_id])

print(len(my_train), len(my_validation))
with open("data/data/my_train.json", "w") as f:
    json.dump(my_train, f, ensure_ascii=False, indent=2)
with open("data/data/my_validation.json", "w") as f:
    json.dump(my_validation, f, ensure_ascii=False, indent=2)
