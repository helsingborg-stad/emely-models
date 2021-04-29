from parlai.scripts.interactive import Interactive
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.core.worlds import validate

from pathlib import Path


def main():
    Interactive.main(
        model_file='zoo:blender/blender_90M/model',
        task='internal'
    )
    return


def own():
    # Load model
    opt = Opt.load(model_opt_path.as_posix())
    opt['task'] = 'internal'
    agent = TransformerGeneratorAgent(opt)

    ################# SET IT TO EVAL #####################

    #
    list_of_text = ['hello i am a transformer']
    list_of_text.append(input('Say something'))


    # ALTERNATIVELY:
    dummy_message = Message()
    dummy_message['id'] = 'localHuman'
    dummy_message['text'] = 'Hi'
    dummy_message['episode_done'] = False
    dummy_message['label_candidates'] = None
    agent.observe(dummy_message)

    for i, text in enumerate(list_of_text):
        message = Message()
        message['text'] = text
        message['episode_done'] = False
        message['label_candidates'] = None

        if i % 2 == 1:
            message['id'] = 'localHuman'
            agent.observe(validate(message))
        else:
            message['id'] = 'TransformerGenerator'
            agent.self_observe(validate(message)) # HM I won't have beam_texts in my message, is that fine????

    agent.act()

    for i, text in enumerate(list_of_text):
        if i != len(list_of_text) -1:
            message = Message()
            message['id'] = 'localHuman' if i % 2 == 0 else 'TransformerGenerator'
            message['episode_done'] = False
            message['text'] = 'HIHIHIHIHIH hello'
            agent.history.update_history(message)
        else:
            observation = Message()
            observation['id'] = 'localHuman'
            observation['episode_done'] = False
            observation['text'] = text
    # while True:


#     message = Message
#    agent.observe(validate(message))
#   response = agentsact()

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    model_opt_path = project_dir / 'ParlAI/data/models/blender/blender_90M/model.opt'
    own()
    #main()
