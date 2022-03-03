from .utils import unique_ner_tags
import json
import os
from tempfile import NamedTemporaryFile

def ner_tags_to_json(training_file,dst_path,tags_field):
    with open(training_file) as json_file:
        data = json.load(json_file)

    training_data = data['TRAINING']
    unique_tags = {}
    all_tags = []
    for doc in training_data:
        tags = unique_ner_tags(doc[tags_field])
        all_tags.extend(tags)
    all_tags = unique_ner_tags(all_tags)
    for i in range(0,len(all_tags)):
        unique_tags[all_tags[i]] = i

    print("The unique NER tags in corpus are: ")
    print(unique_tags)
    with open(os.path.join(dst_path, "ner_tags.json"), "w") as outfile:
        json.dump(unique_tags, outfile, indent = 4)

if __name__ == '__main__':
   ner_tags_to_json()