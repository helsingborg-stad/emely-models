from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, BlenderbotSmallTokenizer, \
    BlenderbotSmallForConditionalGeneration

from pathlib import Path
from argparse import ArgumentParser

""" Used to download huggingface models to /models to avoid caching """

def main(args):
    mname = args.model_name
    model_name = mname.replace('facebook/', '')
    if 'small' in mname:
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
        tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)
    else:
        model = BlenderbotForConditionalGeneration.from_pretrained(mname)
        tokenizer = BlenderbotTokenizer.from_pretrained(mname)

    model_dir = Path(__file__).resolve().parents[2] / 'models' / model_name / 'model'
    token_dir = Path(__file__).resolve().parents[2] / 'models' / model_name / 'tokenizer'

    model_dir.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(token_dir)
    print('Saved model to ', model_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Huggingface model to download')
    args = parser.parse_args()
    main(args)
