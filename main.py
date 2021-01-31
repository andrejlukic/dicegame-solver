import numpy as np
import itertools

from dice_game import DiceGame
from abc import ABC, abstractmethod

DBG = True
def log(msg):
    if DBG:
        print(msg)

class DiceGameAgent(ABC):
    def __init__(self, game):
        self.game = game

    @abstractmethod
    def play(self, state):
        pass


class AlwaysHoldAgent(DiceGameAgent):
    def play(self, state):
        return (0, 1, 2)


class PerfectionistAgent(DiceGameAgent):
    def play(self, state):
        if state == (1, 1, 1) or state == (1, 1, 6):
            return (0, 1, 2)
        else:
            return ()


def play_game_with_agent(agent, game, verbose=False):
    state = game.reset()

    if (verbose): print(f"Testing agent: \n\t{type(agent).__name__}")
    if (verbose): print(f"Starting dice: \n\t{state}\n")

    game_over = False
    actions = 0
    while not game_over:
        action = agent.play(state)
        actions += 1

        if (verbose): print(f"Action {actions}: \t{action}")
        _, state, game_over = game.roll(action)
        if (verbose and not game_over): print(f"Dice: \t\t{state}")

    if (verbose): print(f"\nFinal dice: {state}, score: {game.score}")

    return game.score

def value_iteration(game):
    v_arr = {}
    policy = {}
    for state in game.states:
        v_arr[state] = 0
        policy[state] = ()
    #print(policy)

    gamma = 0.9
    thetta = 0.01
    delta = 1000
    actions = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    log("Starting value iteration gamme = {}, thetta = {}".format(gamma, thetta))
    while (delta >= thetta):
        delta = 0
        for state in game.states:
            s_val = v_arr[state]
            max_action = 0
            for action in actions:
                sum = 0
                states, game_over, reward, probabilities = game.get_next_states(action, state)
                for s1, p1 in zip(states, probabilities):
                    if not game_over:
                        sum += p1 * (reward + gamma * v_arr[s1])
                    else:
                        sum += p1 * (reward + gamma * game.final_score(state))
                if sum > max_action:
                    max_action = sum
                    policy[state] = action
            v_arr[state] = max_action
            delta = max(delta, abs(s_val - v_arr[state]))
        log("Delta = {}".format(delta))
    log("Policy converged = {}".format(policy))

    return policy


class MyAgent(DiceGameAgent):
    def __init__(self, game):
        """
        if your code does any pre-processing on the game, you can do it here

        e.g. you could do the value iteration algorithm here once, store the policy,
        and then use it in the play method

        you can always access the game with self.game
        """
        actions = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
        v_arr = {}
        policy = {}
        for state in game.states:
            v_arr[state] = 0
            policy[state] = ()
        # print(policy)

        gamma = 0.9
        thetta = 0.01
        delta = 1000
        while (delta >= thetta):
            delta = 0
            for state in game.states:
                s_val = v_arr[state]
                max_action = 0
                for action in actions:
                    sum = 0
                    states, game_over, reward, probabilities = game.get_next_states(action, state)
                    for s1, p1 in zip(states, probabilities):
                        if not game_over:
                            sum += p1 * (reward + gamma * v_arr[s1])
                        else:
                            sum += p1 * (reward + gamma * game.final_score(state))
                    if sum > max_action:
                        max_action = sum
                        policy[state] = action
                v_arr[state] = max_action
                delta = max(delta, abs(s_val - v_arr[state]))

        self._policy = policy
        # return policy

        # this calls the superclass constructor (does self.game = game)
        super().__init__(game)

        # YOUR CODE HERE

    def play(self, state):
        """
        given a state, return the chosen action for this state
        at minimum you must support the basic rules: three six-sided fair dice

        if you want to support more rules, use the values inside self.game, e.g.
            the input state will be one of self.game.states
            you must return one of self.game.actions

        read the code in dicegame.py to learn more
        """
        # YOUR CODE HERE

        return self._policy[state]

if __name__ == "__main__":
    # random seed makes the results deterministic
    # change the number to see different results
    #  or delete the line to make it change each time it is run
    #states = np.array(list(itertools.combinations_with_replacement(np.arange(1, game._sides + 1),
    #                                                                   game._dice)),
    #                      dtype=np.int)

    SKIP_TESTS = False

    if not SKIP_TESTS:
        import time



        print("Testing basic rules.")
        print()

        scores = []
        for _ in range(10):
            total_score = 0
            total_time = 0
            n = 1000

            np.random.seed()
            game = DiceGame()

            start_time = time.process_time()
            test_agent = MyAgent(game)
            total_time += time.process_time() - start_time

            for i in range(n):
                start_time = time.process_time()
                score = play_game_with_agent(test_agent, game)
                total_time += time.process_time() - start_time

                print(f"Game {i} score: {score}")
                total_score += score

            scores.append(total_score / n)
            print(f"Average score: {total_score / n}")
            print(f"Total time: {total_time:.4f} seconds")
        print("Overall AVG score {}".format(np.mean(scores)))