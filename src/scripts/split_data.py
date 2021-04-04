import copy
import json

TRAINING_TRACKS = "./data/data/train-tracks_2.json"

with open(TRAINING_TRACKS, "r") as f_r:
    train_tracks = json.load(f_r)
    track_ids = list(train_tracks.keys())

my_train = dict()
my_validation = dict()
for i, track_id in enumerate(track_ids):
    if "S01" in train_tracks[list(train_tracks.keys())[i]]["frames"][0]:
        my_validation[track_id] = copy.deepcopy(train_tracks[track_id])
    else:
        my_train[track_id] = copy.deepcopy(train_tracks[track_id])

print(len(my_train), len(my_validation))
with open("data/data/my_train.json", "w") as f:
    json.dump(my_train, f, ensure_ascii=False, indent=2)
with open("data/data/my_validation.json", "w") as f:
    json.dump(my_validation, f, ensure_ascii=False, indent=2)
