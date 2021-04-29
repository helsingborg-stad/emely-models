import subprocess
import os
from pathlib import Path
import wandb
import datetime
from parlai.scripts.train_model import TrainModel

""" Script for training several models with parlai in a row. All models are validated on the internal task only despite training on different tasks """

small_opts = {'embedding_size': 512,
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

pushshift_config = {'init_model': 'zoo:tutorial_transformer_generator/model',
                    'dict_file': 'zoo:tutorial_transformer_generator/model.dict',
                    }

blender_config = {'init_model': 'zoo:blender/blender_90M/model',
                  'dict_file': 'zoo:blender/blender_90M/model.dict'}

eval_opt_standard = {'task': 'internal',
                     'datatype': 'valid',
                     }


def init_opts():
    """ Merge config dicts """
    pushshift_opts = {**small_opts, **pushshift_config, 'num_epochs': num_epochs}
    blender_opts = {**small_opts, **blender_config, 'num_epochs': num_epochs}
    return pushshift_opts, blender_opts



if __name__ == '__main__':
    # TODO: Add evaluation(display model) to the script
    # TODO: Fix wandb run names

    # Hyperparams
    num_epochs = 100
    os.environ['WANDB_MODE'] = 'dryrun'

    # Project variables
    project_dir = Path(__file__).resolve().parents[2]
    model_run_dir = project_dir / 'model-runs'
    pushshift_opts, blender_opts = init_opts()

    # Timestamp
    now = datetime.datetime.now()
    day_month = str(now.month) + '-' + str(now.day) + '-'

    # What to run in lists
    runs = []

    #####################
    # New training config: pushshift + internal / otter
    #####################
    name = 'pushshift-90M-internal-otter-2-1'
    run_opt = pushshift_opts.copy()
    run_opt['task'] = 'internal,otter'
    run_opt['multitask_weights'] = '2,1'

    # Auto
    run_name = day_month + name
    run_opt['model_file'] = (model_run_dir / run_name / 'model' ).as_posix()

    # Add to list
    eval_opt = make_eval_opt(run_name)
    runs.append(run_opt)
    evals.append(eval_opt)

    #####################
    # New training config: pushshift + internal / otter / bst
    #####################

    name = 'pushshift-90M-internal-otter-bst-2-1-4'
    run_opt = pushshift_opts.copy()
    run_opt['task'] = 'internal,otter,blended_skill_talk'
    run_opt['multitask_weights'] = '2,1,4'

    # Auto
    run_name = day_month + name
    run_opt['model_file'] = (model_run_dir / run_name / 'model' ).as_posix()

    # Add to list
    eval_opt = make_eval_opt(run_name)
    runs.append(run_opt)
    evals.append(eval_opt)

    #####################
    # New training config: blender + internal / otter
    #####################

    name = 'blender-90M-internal-otter-2-1'
    run_opt = blender_opts.copy()
    run_opt['task'] = 'internal,otter'
    run_opt['multitask_weights'] = '2,1'

    # Auto
    run_name = day_month + name
    run_opt['model_file'] = (model_run_dir / run_name / 'model' ).as_posix()

    # Add to list
    eval_opt = make_eval_opt(run_name)
    runs.append(run_opt)
    evals.append(eval_opt)

    #####################
    # New training config: blender + internal / otter / bst
    #####################

    name = 'blender-90M-internal-otter-bst-3-1-1'
    run_opt = blender_opts.copy()
    run_opt['task'] = 'internal,otter,blended_skill_talk'
    run_opt['multitask_weights'] = '3,1,1'

    # Auto
    run_name = day_month + name
    run_opt['model_file'] = (model_run_dir / run_name / 'model' ).as_posix()

    # Add to list
    eval_opt = make_eval_opt(run_name)
    runs.append(run_opt)
    evals.append(eval_opt)



    # Training
    for run_opt in runs:
        # TODO: Replace os.system with TrainMode.main call

        TrainModel.main(**run_opt)

    # # Eval in console and save it to file in checkpoint directory
    # for eval_opt in evals:
        print('Evaluating: ', eval_opt['eval_name'])
        cmd = eval_opt['eval_cmd'].split(' ')
        try:
            cmd.remove('')
        except:
            pass
        eval_dialog = subprocess.check_output(cmd)
        save_eval(eval_dialog, eval_opt['eval_name'])
