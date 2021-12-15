import warnings

from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.core.worlds import validate
from parlai.core.torch_generator_agent import SearchBlocklist
import logging
from typing import List


class EmelyAgent(TransformerGeneratorAgent):
    """ Emely is made to handle several simultaneous dialogs and is deployed on Google Cloud Run.
    - The inference request can jump to a new instance in the middle of a conversation due to GCP scaling up containers
    - Emely needs to be able to build her memory of the dialog from scratch
    """
    
    def __init__(self, opt: Opt, persona=None):
        super().__init__(opt)
        self.persona = persona
        self.persona_is_in_history = False
        self.add_persona_to_history()
        
        
    def add_persona_to_history(self):
        """ Adds the persona to the history to include it during inference """
        if self.persona_is_in_history:
            warnings.warn('Persona is already added to the agent history')
            return
        elif self.persona is None:
            warnings.warn('Tried to add persona to agent history but EmelyAgent.persona is None')
            return
        
        self.observe({'text': self.persona, 'episode_done': False})
        self.persona_is_in_history = True
    

    def observe_and_act(self, text: str) -> str:
        """ Emely gets a string text that she updates her history before she acts """

        # Emely builds her history/memory of the conversation
        self.build_dialog_history(text)

        # Emely acts
        logging.info(msg='Emely has build her history:')
        logging.info(msg=self.history.get_history_str())
        output = self.act()
        reply = output['text']

        # We make emely forget so that she can handle another conversation
        self.amnesia()

        return reply

    
    def build_dialog_history(self, dialogue_string):
        """ Build dialog history from text sent from EmelyBackend """

        # Makes sure history is Empty before building it
        if len(self.history.history_strings) > 0:
            warnings.warn('The history was not empty. Forcing amnesia')
            self.amnesia()
            
        # History starts with her persona
        self.add_persona_to_history()

        list_of_text = dialogue_string.split('\n')

        # The underlying TransformerGeneratorAgent assumes that the user starts all conversations so we have to call a special function in case it was Emely who did
        first_message = Message(text=list_of_text[0],
                                episode_done = False,
                                label_candidates = None)
        if len(list_of_text) % 2 == 0:
            # Even number of messages => Emely Starts the conversation
            first_message['id'] = 'TransformerGenerator'
            self._special_self_observe(first_message)
            observe_who_next = 'localHuman'
            
        else:
            # 
            first_message['id'] = 'localHuman'
            self.observe(first_message)
            observe_who_next = 'TransformerGenerator'

        list_of_text.pop(0)
        # Walk throught the rest of the list of message and observe them 
        for i, text in enumerate(list_of_text):
            message = Message(text=text,
                              episode_done = False,
                              label_candidates = None)

            if observe_who_next == 'localHuman':
                message['id'] = observe_who_next
                self.observe(validate(message))
                observe_who_next = 'TransformerGenerator'
                
            elif observe_who_next == 'TransformerGenerator':
                message['id'] = observe_who_next
                self.self_observe(validate(message))
                observe_who_next = 'localHuman'
            
            else:
                raise ValueError(f'No agent named {observe_who_next}')


    def amnesia(self):
        """ Forget the dialog history and observation """
        self.history.reset()
        self.observation = None
        self.persona_is_in_history = False
        return
    
    
    def _special_self_observe(self, self_message: Message) -> None:
        """
        Observe Emely's first utterance. Overides torch_agent.self_observe()
        Since Emely is designed to start the conversation self.observation will be None at the start
        :param self_message:
            The message.
        """
        use_reply = self.opt.get('use_reply', 'label')

        # quick check everything is in order
        assert len(self.history.history_strings) == 0

        assert self.observation is None

        # We did reply! Safety check is good next round.
        self.__expecting_to_reply = False

        # otherwise, we use the last output the model generated
        if self_message is not None:
            last_reply = self_message['text']
            self.history.add_reply(last_reply)
            return

        raise RuntimeError("Unexpected case in self_observe.")

    
    def set_block_list(self, block_list: List[str]):
        """Sets the beam_block_list for Emely

        Args:
            block_list (List[str]): ngrams to block during generation
        """

        beam_block_list = SearchBlocklist(self.dict)

        try:
            for line in block_list:
                beam_block_list.add(line.strip())
        except IOError:
            logging.error(
                f"Could not load beam block_list. Using empty block_list."
            )
        self.beam_block_list = beam_block_list
        return

        