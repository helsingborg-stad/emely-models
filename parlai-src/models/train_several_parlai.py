import subprocess
import os
from pathlib import Path
import wandb
import datetime
from parlai.scripts.train_model import TrainModel

""" Script for training several models with parlai in a row. All models are validated on the internal task only despite training on different tasks """

pushshift_opts = {'init_model': 'zoo:tutorial_transformer_generator/model',
                  'dict_file': 'zoo:tutorial_transformer_generator/model.dict',
                  'embedding_size': '512',
                  'n_layers': '8',
                  'ffn_size': '2048',
                  'dropout': '0.1',
                  'n_heads': '16',
                  'learn_positional_embeddings': 'True',
                  'n_positions': '512',
                  'variant': 'xlm',
                  'activation': 'gelu',
                  'fp16': 'True',
                  'text_truncate': '512',
                  'label_truncate': '128',
                  'dict_tokenizer': 'bpe',
                  'optimizer': 'adamax',
                  'lr_scheduler': 'reduceonplateau',
                  'betas': '0.9,0.999',
                  'update_freq': '1',
                  'attention_dropout': '0.0',
                  'relu_dropout': '0.0',
                  'model': 'transformer/generator',
                  'dict_lower': 'True',
                  'lr': '1e-06',
                  'gradient_clip': '0.1',
                  'veps': '0.25',
                  'skip_generation': 'False',
                  'vp': '15',
                  'stim': '60',
                  'vme': '20000',
                  'bs': '16',
                  'vmt': 'ppl',
                  'vmm': 'min',
                  'save_after_valid': 'True',
                  'wblog': 'True',
                  'wandb_project': 'parlaiemely',
                  'tensorboard_log': 'True',
                  'metrics': 'ppl,bleu-4,rouge-L',
                  'evaltask': 'internal'}

blender_opts = {'init_model': 'zoo:blender/blender_90M/model',
                'dict_file': 'zoo:blender/blender_90M/model.dict',
                'embedding_size': 512,
                'n_layers': 8,
                'ffn_size': 2048,
                'dropout': 0.1,
                'n_heads': 16,
                'learn_positional_embeddings': True,
                'n_positions': 512,
                'variant': 'xlm',
                'activation': 'gelu',
                'fp16': True,
                'text_truncate': 512,
                'label_truncate': 128,
                'dict_tokenizer': 'bpe',
                'optimizer': 'adamax',
                'lr_scheduler': 'reduceonplateau',
                'betas': '0.9,0.999',
                'update_freq': 1,
                'attention_dropout': 0.0,
                'relu_dropout': 0.0,
                'model': 'transformer/generator',
                'dict_lower': True,
                'lr': 1e-06,
                'gradient_clip': 0.1,
                'veps': 0.25,
                'skip_generation': False,
                'vp': 15,
                'stim': 60,
                'vme': 20000,
                'bs': 16,
                'vmt': 'ppl',
                'vmm': 'min',
                'save_after_valid': True,
                'wblog': True,
                'wandb_project': 'parlaiemely',
                'tensorboard_log': True,
                'metrics': 'ppl,bleu-4,rouge-L',
                'evaltask': 'internal'}

eval_opt = {'task': 'internal',
            'datatype': 'valid',
            }


def make_run_opt(cmd, tasks, weights, name):
    """ Reformats the parlai train cmd with tasks, weights and checkpoint directory
        Returns a dict run_opt
    """
    model_dir = project_dir / 'model-runs/{}/model'.format(name)
    Path.mkdir(Path(model_dir).parent, exist_ok=True, parents=True)

    cmd = cmd.replace('TASKS', tasks)
    cmd = cmd.replace('WEIGHTS', weights)
    cmd = cmd.replace('MODEL_DIR', model_dir.as_posix())
    run_opt = {'run_name': name, 'run_cmd': cmd}
    return run_opt


def make_eval_opt(name):
    """ Formats the display model command for text evaluation of model """
    model_dir = project_dir / 'model-runs/{}/model'.format(name)
    cmd = eval_cmd.replace('CHECKPOINT_DIR', model_dir.as_posix())
    eval_opt = {'eval_name': name, 'eval_cmd': cmd}
    return eval_opt


def save_eval(dialog, name):
    """ Saves the dialog printed during display model to the directory of the corresponding model """
    file_path = project_dir / 'model-runs/{}/display_model.txt'.format(name)
    with open(file_path, 'wb') as f:
        f.write(dialog)
    return


if __name__ == '__main__':
    # TODO: Add evaluation(display model) to the script
    # TODO: Fix wandb run names

    # Hyperparams
    num_epochs = 2
    epochs_cmd = ' --num-epochs {}'.format(num_epochs)

    # Timestamp
    now = datetime.datetime.now()
    day_month = str(now.month) + '-' + str(now.day) + '-'

    # Parlai commands
    pushshift_cmd = """parlai train_model -t TASKS --multitask-weights WEIGHTS -m transformer/generator --init-model zoo:tutorial_transformer_generator/model --dict-file zoo:tutorial_transformer_generator/model.dict --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --fp16 True --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 -vme 20000 -bs 16 -vmt ppl -vmm min --save-after-valid True -wblog True --wandb-project parlaiemely --tensorboard-log --metrics ppl,bleu-4,rouge-L --model-file MODEL_DIR --evaltask internal """
    blender_cmd = """parlai train_model -t TASKS --multitask-weights WEIGHTS -m transformer/generator --init-model zoo:blender/blender_90M/model --dict-file zoo:blender/blender_90M/model.dict --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --fp16 True --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 -vme 20000 -bs 16 -vmt ppl -vmm min --save-after-valid True -wblog True --wandb-project parlaiemely --tensorboard-log --metrics ppl,bleu-4,rouge-L --model-file MODEL_DIR --evaltask internal """
    eval_cmd = """parlai display_model -t internal --datatype valid -mf CHECKPOINT_DIR --skip-generation False"""

    # What to run in lists
    runs = []
    evals = []

    project_dir = Path(__file__).resolve().parents[2]
    model_run_dir = project_dir / 'model-runs'

    # transformer/generator with otter and internal
    run_name = day_month + 'pushshift-90M-internal-otter-2-1'
    run_opt = pushshift_opts.copy()
    run_opt['task'] = 'internal,otter'
    run_opt['multitask_weights'] = '2,1'
    run_opt['model_file'] = model_run_dir / run_name

    # Add to list
    eval_opt = make_eval_opt(run_name)
    runs.append(run_opt)
    evals.append(eval_opt)

    # transformer/generator with otter, internal and blended_skill_talk
    # TODO: FIX THIS WITH THE NEW OPTS
    # run_name = day_month + 'pushshift-90M-internal-otter-bst-1-2-4'
    # tasks = 'internal,otter,blended_skill_talk'
    # weights = '2,1,4'
    # # Add to list
    # run_opt = make_run_opt(pushshift_cmd, tasks, weights, run_name)
    # eval_opt = make_eval_opt(run_name)
    # runs.append(run_opt)
    # evals.append(eval_opt)
    #
    # # blender-90M with internal and otter
    # run_name = day_month + 'blender-90M-internal-otter-2-1'
    # tasks = 'internal,otter'
    # weights = '2,1'
    # # Add to list
    # run_opt = make_run_opt(pushshift_cmd, tasks, weights, run_name)
    # eval_opt = make_eval_opt(run_name)
    # runs.append(run_opt)
    # evals.append(eval_opt)

    # Init Wandb to avoid problems
    os.environ['WANDB_MODE'] = 'dryrun'
    # key = '1eeeef7c0c50125075d2b7a1c5ac2389bd3ae67d'
    # wandb.login(key=key)
    # os.system('wandb ')

    # Training
    for run_opt in runs:
        print('Now running: ', run_opt['run_name'])
        cmd = run_opt['run_cmd'] + epochs_cmd
        os.system(cmd)
        # TODO: Replace os.system with TrainMode.main call

        pushshift_opts = {'init_model': 'zoo:tutorial_transformer_generator/model',
                          'dict_file': 'zoo:tutorial_transformer_generator/model.dict',
                          'embedding_size': '512',
                          'n_layers': '8',
                          'ffn_size': '2048',
                          'dropout': '0.1',
                          'n_heads': '16',
                          'learn_positional_embeddings': 'True',
                          'n_positions': '512',
                          'variant': 'xlm',
                          'activation': 'gelu',
                          'fp16': 'True',
                          'text_truncate': '512',
                          'label_truncate': '128',
                          'dict_tokenizer': 'bpe',
                          'optimizer': 'adamax',
                          'lr_scheduler': 'reduceonplateau',
                          'betas': '0.9,0.999',
                          'update_freq': '1',
                          'attention_dropout': '0.0',
                          'relu_dropout': '0.0',
                          'wandb_project': 'parlaiemely',
                          'tensorboard_log': 'True',
                          'metrics': 'ppl,bleu-4,rouge-L',
                          'model_file': 'MODEL_DIR',
                          'evaltask': 'internal'}

        TrainModel.main(
            # we MUST provide a filename
            model_file='from_scratch_model/model',
            # train on empathetic dialogues
            task='empathetic_dialogues',
            # limit training time to 2 minutes, and a batchsize of 16
            max_train_time=2 * 60,
            batchsize=16,

            # we specify the model type as seq2seq
            model='seq2seq',
            # some hyperparamter choices. We'll use attention. We could use pretrained
            # embeddings too, with embedding_type='fasttext', but they take a long
            # time to download.
            attention='dot',
            # tie the word embeddings of the encoder/decoder/softmax.
            lookuptable='all',
            # truncate text and labels at 64 tokens, for memory and time savings
            truncate=64,
        )

    # # Eval in console and save it to file in checkpoint directory
    for eval_opt in evals:
        print('Evaluating: ', eval_opt['eval_name'])
        cmd = eval_opt['eval_cmd'].split(' ')
        try:
            cmd.remove('')
        except:
            pass
        eval_dialog = subprocess.check_output(cmd)
        save_eval(eval_dialog, eval_opt['eval_name'])
