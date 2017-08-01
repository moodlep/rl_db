import json
import numpy as np
import Environment as env

class User():
    def __init__(self, max_rounds):
        '''User initialised with empty deck, hand, discard lists. 
        Note: in_play is not used yet. 
        hand_stats used in play() to determine what action to follow
        self.env is used to communicate with the environment and get the next state, reward, etc. 
        '''

        # hand_stats: coins, cards, actions, VP, buys, attack, duration:
        self.hand_stats = np.zeros(7)

        # define the mostly internal agent lists used during play
        self.deck = []
        self.hand = []
        self.discard = []
        self.in_play = []

        # Sort out the environment related params:
        self.env = env.DominionEnvironment()
        self.initial_state = self.env.reset()
        self.game_deck = self.prepare_deck()

        self.round = 0
        self.max_rounds = max_rounds
        self.score = 0.0

        # start with 10 cards (7 coppers and 3 estates). implement shuffle and split into deck and hand
        for i in range(7): self.deck.append('copper')
        for i in range(3): self.deck.append('estate')
        self.deal()

        #print some starting stats
        print("hand stats: ", self.hand_stats)
        print("user hand: ", self.hand)
        print("remaining deck cards: ", self.env.card_deck)

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
        print("Round: ", self.round)
        print("hand stats: ", self.hand_stats)
        print("user hand: ", self.hand)
        print("remaining deck cards: ", self.env.card_deck)
        print("discard: ", self.discard)

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
        if self.env.buy_card(card):
            self.discard.append(card)
        else:
            print("error buying card")

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

        return game_cards

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
        print("user hand: ", self.hand)
        print("user deck: ", self.deck)
        print("discard: ", self.discard)
        print("final score is: ", self.score)



# test game

max_rounds = 16
user = User(max_rounds)
print("hand stats: ", user.hand_stats)
print("user hand: ", user.hand)
print("remaining deck cards: ", user.env.card_deck)
user.play()

print("Round: ", user.round)
print("hand stats: ", user.hand_stats)
print("user hand: ", user.hand)
print("remaining deck cards: ", user.env.card_deck)
print("discard: ", user.discard)

for i in range(max_rounds-1):
    user.deal()
    user.play()

    print("Round: ", user.round)
    print("hand stats: ", user.hand_stats)
    print("user hand: ", user.hand)
    print("remaining deck cards: ", user.env.card_deck)
    print("discard: ", user.discard)

# score the game:
print("Final Scores: ")
print("user hand: ", user.hand)
print("user deck: ", user.deck)
print("discard: ", user.discard)
user.final_score()
