import json
import numpy as np

class DominionEnvironment():

    def _extract_decks(self):
        game = {}
        with open('cardsv2.json') as json_file:
            cards = json.load(json_file)
            for card in cards['cards']:
                game[card['name']] = card['count']
        return game, cards

    def _step(self, action):

        # based on action, update card_deck

        # check if game has ended and set flag

        # take a step - play a dummy user

        # based on action, update card_deck

        # check if game has ended and set flag

        # return next state, reward, done flag where state is all cards remaining and if any attacks were implemented.

        return True

    def __init__(self):
        # define the game environment
        # game end flag - signals final round
        self.game_end_flag = False

        # get the deck of cards for this game
        self.card_deck, self.cards = self._extract_decks()

    def get_deck(self):
        return self.card_deck

    def buy_card(self, card):
        # reduce no of cards in self.card_deck
        self.card_deck[card] -=1
        # check if the game end conditions are met and set game end flag

        return True

    def reset(self):
        # starting state returned to all players
        return self.card_deck