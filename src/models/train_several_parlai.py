import os

if __name__ == '__main__':
    # TODO: Add evaluation(display model) to the script
    # TODO: Fix wandb run names

    # Hyperparams
    num_epochs = 100
    epochs_cmd = '--num-epochs {}'.format(num_epochs)

    # What to run in a list
    runs = []

    # Add items as dict to run here
    #   run_cmd = ''
    #   run_name = ''

    # transformer/generator with otter and internal
    run_name = 'pushshift-90M-internal-otter-2-1'
    run_cmd = """parlai train_model -t internal,otter --multitask-weights 2,1 -m transformer/generator --init-model zoo:tutorial_transformer_generator/model --dict-file zoo:tutorial_transformer_generator/model.dict --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --fp16 True --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 -vme 20000 -bs 16 -vmt ppl -vmm min --save-after-valid True -wblog True --wandb-project EmelyModels --metrics ppl,bleu-4,rouge-L --model-file model-runs/SOMEDIR/model """
    run_cmd = run_cmd.replace('SOMEDIR', run_name)
    run_opt = {'run_name': run_name, 'run_cmd': run_cmd}
    runs.append(run_opt)

    # transformer/generator with otter and internal-wpc
    run_name = 'pushshift-90M-internal-wpc-otter-2-1'
    run_cmd = """parlai train_model -t internal-wpc,otter --multitask-weights 2,1 -m transformer/generator --init-model zoo:tutorial_transformer_generator/model --dict-file zoo:tutorial_transformer_generator/model.dict --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --fp16 True --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 -vme 20000 -bs 16 -vmt ppl -vmm min --save-after-valid True -wblog True --wandb-project EmelyModels --metrics ppl,bleu-4,rouge-L --model-file model-runs/SOMEDIR/model """
    run_cmd = run_cmd.replace('SOMEDIR', run_name)
    run_opt = {'run_name': run_name, 'run_cmd': run_cmd}
    runs.append(run_opt)

    # transformer/generator with otter and internal
    run_name = 'pushshift-90M-internal-nhq-otter-2-1'
    run_cmd = """parlai train_model -t internal-nhq,otter --multitask-weights 2,1 -m transformer/generator --init-model zoo:tutorial_transformer_generator/model --dict-file zoo:tutorial_transformer_generator/model.dict --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --fp16 True --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 -vme 20000 -bs 16 -vmt ppl -vmm min --save-after-valid True -wblog True --wandb-project EmelyModels --metrics ppl,bleu-4,rouge-L --model-file model-runs/SOMEDIR/model """
    run_cmd = run_cmd.replace('SOMEDIR', run_name)
    run_opt = {'run_name': run_name, 'run_cmd': run_cmd}
    runs.append(run_opt)

    # blender-90M with internal and otter
    run_name = 'blender-90M-internal-otter-2-1'
    run_cmd = """parlai train_model -t internal,otter --multitask-weights 2,1 -m transformer/generator --init-model zoo:blender/blender_90M/model --dict-file zoo:blender/blender_90M/model.dict --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --fp16 True --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 -vme 20000 -bs 16 -vmt ppl -vmm min --save-after-valid True -wblog True --wandb-project EmelyModels --metrics ppl,bleu-4,rouge-L --model-file model-runs/SOMEDIR/model """
    run_cmd = run_cmd.replace('SOMEDIR', run_name)
    run_opt = {'run_name': run_name, 'run_cmd': run_cmd}
    runs.append(run_opt)

    # blender-90M with internal and otter
    run_name = 'blender-90M-internal-nhq-otter-2-1'
    run_cmd = """parlai train_model -t internal-nhq,otter --multitask-weights 2,1 -m transformer/generator --init-model zoo:blender/blender_90M/model --dict-file zoo:blender/blender_90M/model.dict --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --fp16 True --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 -vme 20000 -bs 16 -vmt ppl -vmm min --save-after-valid True -wblog True --wandb-project EmelyModels --metrics ppl,bleu-4,rouge-L --model-file model-runs/SOMEDIR/model """
    run_cmd = run_cmd.replace('SOMEDIR', run_name)
    run_opt = {'run_name': run_name, 'run_cmd': run_cmd}
    runs.append(run_opt)

    # blender-90M with internal and otter
    run_name = 'blender-90M-internal-wpc-otter-2-1'
    run_cmd = """parlai train_model -t internal-wpc,otter --multitask-weights 2,1 -m transformer/generator --init-model zoo:blender/blender_90M/model --dict-file zoo:blender/blender_90M/model.dict --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --fp16 True --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 -vme 20000 -bs 16 -vmt ppl -vmm min --save-after-valid True -wblog True --wandb-project EmelyModels --metrics ppl,bleu-4,rouge-L --model-file model-runs/SOMEDIR/model """
    run_cmd = run_cmd.replace('SOMEDIR', run_name)
    run_opt = {'run_name': run_name, 'run_cmd': run_cmd}
    runs.append(run_opt)


    # Actual training and eval
    for run_opt in runs:
        cmd = run_opt['run_cmd'] + epochs_cmd
        os.system(cmd)
