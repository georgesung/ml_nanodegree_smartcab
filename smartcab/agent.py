import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.qtable = {}  # Q-value table: key = state, value = Q-value
        self.prev_sa = None  # keep track of previous state, action pair

        # Constants
        self.SIGMOID_OFFSET = 6
        self.SIGMOID_RATE = 0.05
        self.MIN_RAND_PROB = 0.2
        self.ALPHA = 0.1
        self.GAMMA = 0.7

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        # First, update inputs such that None becomes 'None' (hard to use None as dict key)
        for key in inputs:
            if inputs[key] is None:
                inputs[key] = 'None'

        # Update state
        state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])

        # If Q-value does not exist for current state (any action), make initial Q-value = 0
        valid_actions = ['None', 'forward', 'left', 'right']
        if (state, 'None') not in self.qtable:
            for action in valid_actions:
                self.qtable[(state, action)] = 0

        # TODO: Select action according to your policy
        # Choose between the following two policies randomly, choosing the random policy with decreasing probability as t increases
        # I.e. Sshifting from "exploration" to "exploitation"
        #   * Random policy (minimum probability of choosing random policy is MIN_RAND_PROB)
        #   * Given our current state, choose the action with maximum Q(state, action) value
        prob_q = 1/(1 + math.exp(-self.SIGMOID_RATE*t + self.SIGMOID_OFFSET))  # sigmoid function
        threshold = random.uniform(0, 1)

        if prob_q - self.MIN_RAND_PROB >= threshold:
            qs = [self.qtable[(state, 'None')], self.qtable[(state, 'forward')], self.qtable[(state, 'left')], self.qtable[(state, 'right')]]
            action = valid_actions[qs.index(max(qs))]
        else:
            action = random.choice(valid_actions)

        # Execute action and get reward
        action_out = None if action is 'None' else action
        reward = self.env.act(self, action_out)

        # TODO: Learn policy based on state, action, reward
        # Update the Q-value for the *previous* state, using current reward
        if self.prev_sa is not None:
            new_q = reward + self.GAMMA * max([self.qtable[(self.prev_sa[0], a)] for a in valid_actions])
            self.qtable[self.prev_sa] = (1 - self.ALPHA) * self.qtable[self.prev_sa] + self.ALPHA * new_q

        self.prev_sa = (state, action)

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        print 't = ' + str(t); print self.qtable


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
