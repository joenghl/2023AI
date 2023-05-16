import random, math, os
import numpy as np
from enum import Enum

"""
Enum with the actions that the agent can execute
"""


class Actions(Enum):
    up = 0  # move up
    right = 1  # move right
    down = 2  # move down
    left = 3  # move left
    none = 4  # none or pick
    drop = 5


"""
The class `Game` has been modified.
Game = World (transition model) + RM (reward function)
"""


class Game:
    def __init__(self, task):
        self.world = OfficeWorld()
        self.num_features = len(self.world.get_features())
        self.num_actions = len(self.world.get_actions())
        assert task in ['to_a', 'to_b', 'to_c', 'to_d',
                        'get_mail', 'get_coffee', 'to_office']
        self.task = task  # the name of task
        self.game_over = False


    def reset(self):
        self.world = OfficeWorld()
        self.game_over = False
        return self.world.get_features()

    def step(self, a):
        """
        We execute 'action' in the game
        Returns the reward
        """
        assert not self.game_over
        s1 = self.world.get_features()  # current state
        self.world.execute_action(a)  # the state of `world` is modified
        s2 = self.world.get_features()  # next state
        events = self.world.get_true_propositions()
        prob_meaning = self.world.prop_meaning
        if prob_meaning[events] == self.task:
            r = 1
        elif prob_meaning[events] == 'plant':
            r = -1
        else:
            r = -0.01
        done = prob_meaning[events] in ['plant', self.task]
        self.game_over = done
        return s2, r, done, prob_meaning[events]

    def render(self):
        self.world.show()


class OfficeWorld:
    def __init__(self):
        self._load_map()
        self.prop_meaning = {'a': 'to_a', 'b': 'to_b', 'c': 'to_c', 'd': 'to_d',
                             'e': 'get_mail', 'f': 'get_coffee', 'g': 'to_office', 'n': 'plant', '': 'None'}

    def execute_action(self, a):
        """
        We execute 'action' in the game
        """
        action = Actions(a)
        x, y = self.agent
        # executing action
        if (x, y, action) not in self.forbidden_transitions:
            if action == Actions.up: y += 1
            if action == Actions.down: y -= 1
            if action == Actions.left: x -= 1
            if action == Actions.right: x += 1
        self.agent = (x, y)

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.actions

    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = ""
        if self.agent in self.objects:
            ret += self.objects[self.agent]
        return ret

    def get_state(self):
        return None  # we are only using "simple reward machines" for the craft domain

    # The following methods return different feature representations of the map ------------
    def get_features(self):
        x, y = self.agent
        N, M = 12, 9
        ret = np.zeros((N, M), dtype=np.float64)
        ret[x, y] = 1
        return ret.ravel()  # from 2D to 1D (use a.flatten() is you want to copy the array)

    def show(self):
        for y in range(8, -1, -1):
            if y % 3 == 2:
                for x in range(12):
                    if x % 3 == 0:
                        print("_", end="")
                        if 0 < x < 11:
                            print("_", end="")
                    if (x, y, Actions.up) in self.forbidden_transitions:
                        print("_", end="")
                    else:
                        print(" ", end="")
                print()
            for x in range(12):
                if (x, y, Actions.left) in self.forbidden_transitions:
                    print("|", end="")
                elif x % 3 == 0:
                    print(" ", end="")
                if (x, y) == self.agent:
                    print("A", end="")
                elif (x, y) in self.objects:
                    print(self.objects[(x, y)], end="")
                else:
                    print(" ", end="")
                if (x, y, Actions.right) in self.forbidden_transitions:
                    print("|", end="")
                elif x % 3 == 2:
                    print(" ", end="")
            print()
            if y % 3 == 0:
                for x in range(12):
                    if x % 3 == 0:
                        print("_", end="")
                        if 0 < x < 11:
                            print("_", end="")
                    if (x, y, Actions.down) in self.forbidden_transitions:
                        print("_", end="")
                    else:
                        print(" ", end="")
                print()

                # The following methods create the map ----------------------------------------------

    def _load_map(self):
        # Creating the map
        self.objects = {}
        self.objects[(1, 1)] = "a"
        self.objects[(10, 1)] = "b"
        self.objects[(10, 7)] = "c"
        self.objects[(1, 7)] = "d"
        self.objects[(7, 4)] = "e"  # MAIL
        self.objects[(8, 2)] = "f"  # COFFEE
        self.objects[(3, 6)] = "f"  # COFFEE
        self.objects[(4, 4)] = "g"  # OFFICE
        self.objects[(4, 1)] = "n"  # PLANT
        self.objects[(7, 1)] = "n"  # PLANT
        self.objects[(4, 7)] = "n"  # PLANT
        self.objects[(7, 7)] = "n"  # PLANT
        self.objects[(1, 4)] = "n"  # PLANT
        self.objects[(10, 4)] = "n"  # PLANT
        # Adding walls
        self.forbidden_transitions = set()
        # general grid
        for x in range(12):
            for y in [0, 3, 6]:
                self.forbidden_transitions.add((x, y, Actions.down))
                self.forbidden_transitions.add((x, y + 2, Actions.up))
        for y in range(9):
            for x in [0, 3, 6, 9]:
                self.forbidden_transitions.add((x, y, Actions.left))
                self.forbidden_transitions.add((x + 2, y, Actions.right))
        # adding 'doors'
        for y in [1, 7]:
            for x in [2, 5, 8]:
                self.forbidden_transitions.remove((x, y, Actions.right))
                self.forbidden_transitions.remove((x + 1, y, Actions.left))
        for x in [1, 4, 7, 10]:
            self.forbidden_transitions.remove((x, 5, Actions.up))
            self.forbidden_transitions.remove((x, 6, Actions.down))
        for x in [1, 10]:
            self.forbidden_transitions.remove((x, 2, Actions.up))
            self.forbidden_transitions.remove((x, 3, Actions.down))
        # Adding the agent
        self.agent = (2, 1)
        self.actions = [Actions.up.value, Actions.right.value, Actions.down.value, Actions.left.value]


def play(task):
    # commands
    str_to_action = {"w": Actions.up.value, "d": Actions.right.value, "s": Actions.down.value, "a": Actions.left.value}

    # play the game!
    print("Running", task)

    game = Game(task)  # setting the environment
    s1 = game.reset()
    events = ''
    while True:
        # Showing game
        game.render()
        print("Events:", game.world.prop_meaning[events])
        # Getting action
        print("Action? ", end="")
        a = input()
        print()
        # Executing action
        if a in str_to_action:
            s2, r, done, events = game.step(str_to_action[a])
            print("---------------------")
            print("Rewards:", r)
            print("---------------------")

            if done:  # Game Over
                break
            s1 = s2
        else:
            print("Forbidden action")
    game.render()
    print("Events:", events)


# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    play('to_office')
