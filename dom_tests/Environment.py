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

    def step(self, action):

        immediate_reward = 0.0

        # based on action, update card_deck
        for i, card in enumerate(action):
            if card == 1.0:
                immediate_reward += self.buy_card(self.list_of_cards[i])

        # check if game has ended and set flag

        # take a step - play a dummy user

        # return next state, reward, done flag where state is all cards remaining and if any attacks were implemented.

        return self.state, immediate_reward, False

    def __init__(self):
        # define the game environment
        # game end flag - signals final round
        self.game_end_flag = False

        # get the deck of cards for this game
        self.card_deck, self.cards = self._extract_decks()
        self.game_deck, self.list_of_cards, self.count_of_cards = self.prepare_deck()

        # define the state, action and reward variables:
        self.action = np.zeros(np.size(self.count_of_cards), dtype=float)
        self.state = self.count_of_cards

    def prepare_deck(self):
        all_cards = self.cards['cards']

        # Now just extract the cards for this current game deck:
        subset = [(card['name'], card['features']) for card in all_cards if card['name'] in self.card_deck]
        game_cards = dict(subset)
        list_of_cards = list(self.card_deck.keys())
        count_of_cards = np.array(list(self.card_deck.values()), dtype=float)

        return game_cards, list_of_cards, count_of_cards

    def buy_card(self, card):
        reward = 0.0
        # reduce no of cards in self.card_deck
        self.card_deck[card] -=1
        self.state[self.list_of_cards.index(card)] -=1

        # return the VP attached to the card
        if self.game_deck[card].get('VP') is not None:
            reward = self.game_deck[card].get('VP')
        # check if the game end conditions are met and set game end flag

        return reward

    def reset(self):
        # starting state returned to all players
        return self.state


env = DominionEnvironment()
print(env.list_of_cards)
print(env.count_of_cards)
print(env.game_deck)
print(env.card_deck)
