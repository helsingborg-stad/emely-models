import warnings

from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.message import Message
from parlai.core.worlds import validate
import logging



""" Emely extends the TransformerGeneratorAgent"""

class EmelyAgent(TransformerGeneratorAgent):

    def observe_and_act(self, text: str) -> str:
        """ Emely gets a string text that she updates her history before she acts """

        # Emely builds her history/memory of the conversation
        self.build_emely_history(text)

        # Emely acts
        logging.info(msg='Emely has build her history:')
        logging.info(msg=self.history.get_history_str())
        output = self.act()
        reply = output['text']

        # We make emely forget so that she can handle another conversation
        self.amnesia()

        return reply

    def build_emely_history(self, text):
        """ Build dialog history from text sent from EmelyBackend """

        # TODO: CHECK THAT THE HISTORY IS EMPTY
        if len(self.history.history_strings):
            warnings.warn('The history wasn not empty. Forcing amnesia')
            self.amnesia()

        list_of_text = text.split('\n')
        try:
            assert len(list_of_text) % 2 == 0
        except AssertionError:
            list_of_text.pop(0)

        # Walk throught the list of text and
        for i, text in enumerate(list_of_text):
            message = Message()
            message['text'] = text
            message['episode_done'] = False
            message['label_candidates'] = None

            # The underlying TransformerGeneratorAgent assumes that it the user starts all conversations so we have to add a dummy message
            # TODO: Change EmelyAgent to change this
            dummy_message = Message()
            dummy_message['id'] = 'localHuman'
            dummy_message['text'] = 'Hi'
            dummy_message['episode_done'] = False
            dummy_message['label_candidates'] = None
            self.observe(dummy_message)

            # Emely always starts the conversations so we can use i % 2 to know whose message it is
            if i % 2 == 1:
                message['id'] = 'localHuman'
                self.observe(validate(message))
            else:
                message['id'] = 'TransformerGenerator'
                self.self_observe(validate(message))



    def amnesia(self):
        """ Forget entire history """
        self.history.reset()
        return