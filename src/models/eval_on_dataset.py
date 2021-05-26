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

    # Let's save them in two places
    file_path1 = project_dir / 'models/runs/{}/display_model_{}.txt'.format(name, datatype)
    with open(file_path1, 'w') as f:
        f.write(text)


    file_path2 = project_dir / 'models/runs/display-model/{}-{}'.format(name, datatype)
    file_path2.parent.mkdir(exist_ok=True)
    with open(file_path2, 'w') as f:
        f.write(text)


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    model_dir = project_dir / 'models/runs'

    eval_cmd = 'parlai display_model -mf {} -t fromfile:parlaiformat --fromfile-datapath {}'
    datatypes = ['valid', 'train']
    internal_path = project_dir / 'ParlAI/data/internal'

    for dir in model_dir.iterdir():
        if dir.is_dir():
            model_file =  dir / 'model'
            if model_file.exists():
                print('Evaluating: ', dir.name)

                for datatype in datatypes:
                    data_file = internal_path.as_posix() + '/' + datatype + '.txt'
                    cmd = eval_cmd.format(model_file.as_posix(), data_file)
                    cmd_list = cmd.split(' ')
                    eval_dialog = subprocess.check_output(cmd_list)
                    save_eval(eval_dialog, dir.name, datatype)
