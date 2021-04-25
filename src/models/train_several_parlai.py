import subprocess
import os
from pathlib import Path

""" Script for training several models with parlai in a row. All models are validated on the internal task only despite training on different tasks """


def make_run_opt(cmd, tasks, weights, name):
    """ Reformats the parlai train cmd with tasks, weights and checkpoint directory
        Returns a dict run_opt
    """
    model_dir = project_dir / 'model-runs/{}/model'.format(name)

    cmd = cmd.replace('TASKS', tasks)
    cmd = cmd.replace('WEIGHTS', weights)
    cmd = cmd.replace('MODEL_DIR', model_dir)
    run_opt = {'run_name': name, 'run_cmd': cmd}
    return run_opt


def make_eval_opt(name):
    """ Formats the display model command for text evaluation of model """
    model_dir = project_dir / 'model-runs/{}/model'.format(name)
    cmd = eval_cmd.replace('CHECKPOINT_DIR', model_dir)
    eval_opt = {'eval_name': name, 'eval_cmd': cmd}
    return eval_opt


def save_eval(dialog, name):
    """ Saves the dialog printed during display model to the directory of the corresponding model """
    file_path = project_dir / 'model-runs/{}/display_model.txt'.format(name)
    with open(file_path, 'w') as f:
        f.write(dialog)
    return


if __name__ == '__main__':
    # TODO: Add evaluation(display model) to the script
    # TODO: Fix wandb run names

    # Hyperparams
    num_epochs = 100
    epochs_cmd = '--num-epochs {}'.format(num_epochs)

    # Parlai commands
    pushshift_cmd = """parlai train_model -t TASKS --multitask-weights WEIGHTS -m transformer/generator --init-model zoo:tutorial_transformer_generator/model --dict-file zoo:tutorial_transformer_generator/model.dict --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --fp16 True --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 -vme 20000 -bs 16 -vmt ppl -vmm min --save-after-valid True -wblog True --wandb-project EmelyModels --metrics ppl,bleu-4,rouge-L --model-file MODEL_DIR --evaltask internal"""
    blender_cmd = """parlai train_model -t TASKS --multitask-weights WEIGHTS -m transformer/generator --init-model zoo:blender/blender_90M/model --dict-file zoo:blender/blender_90M/model.dict --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --fp16 True --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 -vme 20000 -bs 16 -vmt ppl -vmm min --save-after-valid True -wblog True --wandb-project EmelyModels --metrics ppl,bleu-4,rouge-L --model-file MODEL_DIR --evaltask internal"""
    eval_cmd = """parlai display_model -t internal --datatype valid -mf CHECKPOINT_DIR """

    # What to run in lists
    runs = []
    evals = []

    project_dir = Path(__file__).resolve().parents[2]

    # transformer/generator with otter and internal
    run_name = 'pushshift-90M-internal-otter-2-1'
    tasks = 'internal,otter'
    weights = '2,1'
    # Add to list
    run_opt = make_run_opt(pushshift_cmd, tasks, weights, run_name)
    eval_opt = make_eval_opt(run_name)
    runs.append(run_opt)
    evals.append(eval_opt)

    # transformer/generator with otter, internal and blended_skill_talk
    run_name = 'pushshift-90M-internal-otter-bst-1-2-4'
    tasks = 'internal,otter,blended_skill_talk'
    weights = '2,1,4'
    # Add to list
    run_opt = make_run_opt(pushshift_cmd, tasks, weights, run_name)
    eval_opt = make_eval_opt(run_name)
    runs.append(run_opt)
    evals.append(eval_opt)

    # blender-90M with internal and otter
    run_name = 'blender-90M-internal-otter-2-1'
    tasks = 'internal,otter'
    weights = '2,1'
    # Add to list
    run_opt = make_run_opt(pushshift_cmd, tasks, weights, run_name)
    eval_opt = make_eval_opt(run_name)
    runs.append(run_opt)
    evals.append(eval_opt)

    # Training
    for run_opt in runs:
        print('Now running: ', run_opt['run_name'])
        cmd = run_opt['run_cmd'] + epochs_cmd
        os.system(cmd)

    # Eval in console and save it to file in checkpoint directory
    for eval_opt in evals:
        print(eval_opt['eval_name'])
        cmd = eval_opt['eval_cmd']
        eval_dialog = subprocess.check_output(cmd)
        save_eval(eval_dialog, eval_opt['eval_name'])
