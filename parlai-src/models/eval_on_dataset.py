from pathlib import Path
import subprocess
import os

def save_eval(dialog, name, datatype):
    """ Saves the dialog printed during display model to the directory of the corresponding model
        We split on 'NEW EPISODE' and don't include the first bit since it's just model opts
    """
    dialog = dialog.decode("utf-8")
    episodes = dialog.split('NEW EPISODE')
    text = ''
    for i in range(len(episodes) - 1):
        text = text + 'NEW EPISODE '+ episodes[i + 1] + '\n\n'

    file_path = project_dir / 'model-runs/{}/display_model_{}.txt'.format(name, datatype)
    with open(file_path, 'w') as f:
        f.write(text)


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    model_dir = project_dir / 'model-runs'

    eval_cmd = 'parlai display_model -t internal -mf {} --datatype {} --skip-generation False'
    datatypes = ['valid', 'train']

    for dir in model_dir.iterdir():
        if dir.is_dir():
            model_file =  dir / 'model'
            if model_file.exists():
                print('Evaluating: ', dir.name)

                for datatype in datatypes:
                    cmd = eval_cmd.format(model_file.as_posix(), datatype)
                    cmd_list = cmd.split(' ')
                    eval_dialog = subprocess.check_output(cmd_list)
                    save_eval(eval_dialog, dir.name, datatype)
