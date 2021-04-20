from src.models.model import LitBlenderbot
from pathlib import Path
from transformers import BlenderbotTokenizer, BlenderbotSmallTokenizer
from argparse import Namespace, ArgumentParser

"""" Loads a pytorch lightning checkpoint from training and saves tokenizer and model 
     as bin in same directory under /models so it can be used for Emely
"""

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='directory with pytorch lightning checkpoint')
    parser.add_argument('--checkpoint', type=str, required=False, default=None,
                        help='Optional: checkpoint to convert to bin')
    params = parser.parse_args()

    # Model run and checkpoint name
    model_zoo_dir = Path(__file__).resolve().parents[2] / 'models'
    model_dir = model_zoo_dir / params.model_dir
    assert model_dir.exists(), model_dir

    if params.checkpoint is None:  # We look for the latest checkpoint
        checkpoints = [file.name for file in model_dir.iterdir() if file.is_file()]
        checkpoints = sorted(checkpoints, key=lambda x: x[6:9])
        checkpoint = checkpoints[-1]
    else:
        checkpoint = params.checkpoint

    checkpoint_dir = model_dir / checkpoint

    if 'small' in params.model_dir:
        mname = 'blenderbot_small-90M'
        tokenizer_dir = model_zoo_dir / mname / 'tokenizer'
        tokenizer = BlenderbotSmallTokenizer.from_pretrained(tokenizer_dir)
    else:
        mname = params.model_dir.split('@')[0]
        tokenizer_dir = model_zoo_dir / mname / 'tokenizer'
        tokenizer = BlenderbotTokenizer.from_pretrained(tokenizer_dir)

    kwargs = {'mname': mname, 'tokenizer': tokenizer, 'hparams': Namespace(unfreeze_decoder=False)}
    litmodel = LitBlenderbot.load_from_checkpoint(checkpoint_path=checkpoint_dir, **kwargs)
    litmodel.model.save_pretrained(model_dir / 'model')
    litmodel.tokenizer.save_pretrained(model_dir / 'tokenizer')
