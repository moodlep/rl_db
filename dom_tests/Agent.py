import json
import numpy as np
import Environment

class User():
    def __init__(self, max_rounds, env, name, print_debug=False):
        '''User initialised with empty deck, hand, discard lists. 
        Note: in_play is not used yet. 
        hand_stats used in play() to determine what action to follow
        self.env is used to communicate with the environment and get the next state, reward, etc. 
        '''
        self.name = name
        self.print_debug = print_debug

        # hand_stats: coins, cards, actions, VP, buys, attack, duration:
        self.hand_stats = np.zeros(7)

        # define the mostly internal agent lists used during play
        self.deck = []
        self.hand = []
        self.discard = []
        self.in_play = []

        # Sort out the environment related params:
        self.env = env
        self.game_deck, self.list_of_cards, self.count_of_cards = self.prepare_deck()

        # define the state, action and reward variables:
        self.state = self.env.reset()
        self.action = np.zeros(np.size(self.count_of_cards), dtype=float)

        self.round = 0
        self.max_rounds = max_rounds
        self.score = 0.0 # end of game reward
        self.immediate_reward = 0.0 # immediate reward

        # start with 10 cards (7 coppers and 3 estates). implement shuffle and split into deck and hand
        for i in range(7): self.deck.append('copper')
        for i in range(3): self.deck.append('estate')
        self.deal()

        #print some starting stats
        print(self.name, ": ")
        if self.print_debug:
            print(self.name, " hand stats: ", self.hand_stats)
            print(self.name, " user hand: ", self.hand)
            print(self.name, " remaining deck cards: ", self.env.card_deck)

    def play(self):
        # Search list of cards for action cards
        # If found,
        # 	Move card into in_play list
        # 	play_action_card(card, in_play, discard) is called.
        # 	Update_stats(): update the available money in hand
        #
        # Buy(coins, no_of_buys, turn_number):
        # 	If turn_number < halfway point:
        # 		Buy gold if >= 6
        # 		Buy action card (passed in card that is being assessed)
        # 	Else
        # 		Buy VP card/s

        # get the hand_stats
        self.get_hand_stats()

        # buy something
        if self.round < (self.max_rounds/2):
            # buy money
            if self.hand_stats[0] >= 6:
                self.buy_card("gold")
            elif self.hand_stats[0] >= 3:
                self.buy_card("silver")
        else:
            # buy VPs
            if self.hand_stats[0] >= 8:
                self.buy_card("province")
            elif self.hand_stats[0] >= 5:
                self.buy_card("duchy")

        # print some stats:
        print(self.name, " Round: ", self.round)
        print(self.name, " Action: " , self.action)
        if self.print_debug:
            print(self.name, " hand stats: ", self.hand_stats)
            print(self.name, " user hand: ", self.hand)
            print(self.name, " remaining deck cards: ", self.env.card_deck)
            print(self.name, " discard: ", self.discard)

        self.round += 1  # increment the turn/round number

        return True

    '''Utility methods: 
    buy_card: one of the key actions that needs to sync with the environment so the appropriate deck is removed. 
    deal: deal a hand of 5 cards from the deck. If deck empty re-shuffle the discard into the deck. 
    shuffle_discard: supports deal() by refreshing the deck from the discard pile. 
    get_hand_stats: totals the available features of the hand: coin count, action count, etc. 
    prepare_deck: extract all the cards that are appropriate for this game. Call at the start of the game
    final_score: tallies the VPs at the end of the game. 
    '''
    def buy_card(self, card):
        # modify this to take an action
        # if self.env.buy_card(card):
        #     self.discard.append(card)
        # else:
        #     print("error buying card")

        # take an action
        self.action.fill(0.0)
        self.action[self.list_of_cards.index(card)] = 1.0
        self.env.step(self.action)
        self.discard.append(card)

    def deal(self):
        if np.size(self.hand) > 0:
            # first clear out the old hand into the discard
            for card in self.hand:
                self.discard.append(card)
            self.hand.clear()

        for r in range(5):
            deck_size = np.size(self.deck)
            if deck_size > 1:
                self.hand.append(self.deck.pop(np.random.randint(1,np.size(self.deck))))
            elif deck_size == 1:
                self.hand.append(self.deck.pop())
            else:
                self.shuffle_discard()
                self.hand.append(self.deck.pop(np.random.randint(1,np.size(self.deck))))

        # reset the hand_stats
        self.hand_stats = np.zeros(7)

    def shuffle_discard(self):
        self.deck = self.discard.copy()
        self.discard.clear()

    def get_hand_stats(self):
        #hand_stats: coins, cards, actions, VP, buys, attack, duration
        for card in self.hand:
            if self.game_deck[card].get('coins') is not None: self.hand_stats[0] += float(self.game_deck[card].get('coins'))
            if self.game_deck[card].get('cards') is not None: self.hand_stats[1] += float(self.game_deck[card].get('cards'))
            if self.game_deck[card].get('actions') is not None: self.hand_stats[2] += float(self.game_deck[card].get('actions'))
            if self.game_deck[card].get('VP') is not None: self.hand_stats[3] += float(self.game_deck[card].get('VP'))
            if self.game_deck[card].get('buys') is not None: self.hand_stats[4] += float(self.game_deck[card].get('buys'))
            if self.game_deck[card].get('attack') is not None: self.hand_stats[5] += float(self.game_deck[card].get('attack'))
            if self.game_deck[card].get('duration') is not None: self.hand_stats[6] += float(self.game_deck[card].get('duration'))

    def prepare_deck(self):

        all_cards = self.env.cards['cards']

        # Now just extract the cards for this current game deck:
        subset = [(card['name'], card['features']) for card in all_cards if card['name'] in self.env.card_deck]
        game_cards = dict(subset)
        list_of_cards = list(self.env.card_deck.keys())
        count_of_cards = np.array(list(self.env.card_deck.values()), dtype=float)

        return game_cards, list_of_cards, count_of_cards

    def final_score(self):
        #collect scores from deck, hand and discard
        for card in self.deck:
            if self.game_deck[card].get('VP') is not None:
                self.score += float(self.game_deck[card].get('VP'))
        for card in self.hand:
            if self.game_deck[card].get('VP') is not None:
                self.score += float(self.game_deck[card].get('VP'))
        for card in self.discard:
            if self.game_deck[card].get('VP') is not None:
                self.score += float(self.game_deck[card].get('VP'))
        if self.print_debug:
            print(self.name, " user hand: ", self.hand)
            print(self.name, " user deck: ", self.deck)
            print(self.name, " discard: ", self.discard)
        print(self.name, " final score is: ", self.score)


# test game

# max_rounds = 16
# env = Environment.DominionEnvironment()
# user = User(max_rounds, env, "userA")
# user.play()
#
# for i in range(max_rounds-1):
#     user.deal()
#     user.play()
#
# # score the game:
# user.final_score()
