from allennlp_models.tagging.dataset_readers import OntonotesNamedEntityRecognition
from argparse import ArgumentParser

def allennlp_parsing():
    parser = ArgumentParser()
    parser.add_argument(
        '-s',
        '--src',
        dest='source_folder', type=str, required=True,
        help='The source folder with extracted Ontonotes 5 dataset (see '
             'https://catalog.ldc.upenn.edu/LDC2013T19).'
    )

    cmd_args = parser.parse_args()

    onto_parser = OntonotesNamedEntityRecognition()
    instances = list(onto_parser._read(cmd_args.source_folder))
    
    print(instances[0])

if __name__ == '__main__':
    allennlp_parsing()