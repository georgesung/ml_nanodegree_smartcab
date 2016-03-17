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
        self.qtable = {}  # Q-value table: key = state-action, value = Q-value
        self.prev_sa = None  # keep track of previous state-action

        # Constants
        self.INITIAL_Q = 0.
        self.SIGMOID_OFFSET = 6
        self.SIGMOID_RATE = 0.05
        self.MIN_RAND_PROB = 0.2
        self.ALPHA = 1.
        self.GAMMA = 0.

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])

        # If Q-value does not exist for current state (any action), make initial Q-value = 0
        valid_actions = [None, 'forward', 'left', 'right']
        for action in valid_actions:
            if self.compress_sa(state, action) not in self.qtable:
                self.qtable[self.compress_sa(state, action)] = self.INITIAL_Q

        # TODO: Select action according to your policy
        # Choose between the following two policies randomly, choosing the random policy with decreasing probability as t increases
        # I.e. Sshifting from "exploration" to "exploitation"
        #   * Random policy (minimum probability of choosing random policy is MIN_RAND_PROB)
        #   * Given our current state, choose the action with maximum Q(state, action) value
        prob_q = 1/(1 + math.exp(-self.SIGMOID_RATE*t + self.SIGMOID_OFFSET))  # sigmoid function
        threshold = random.uniform(0, 1)

        if prob_q - self.MIN_RAND_PROB >= threshold:
            qs = [self.qtable[self.compress_sa(state, None)], self.qtable[self.compress_sa(state, 'forward')], self.qtable[self.compress_sa(state, 'left')], self.qtable[self.compress_sa(state, 'right')]]
            action = valid_actions[qs.index(max(qs))]
        else:
            action = random.choice(valid_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # DEBUG
        if reward is 2:
            print 'DEBUG: ' + str(state) + ', ' + str(action)
            pass

        # TODO: Learn policy based on state, action, reward
        '''
        # Update the Q-value for the *previous* state, using current reward
        if self.prev_sa is not None:
            new_q = reward + self.GAMMA * max([self.qtable[self.compress_sa(state, a)] for a in valid_actions])
            self.qtable[self.prev_sa] = (1 - self.ALPHA) * self.qtable[self.prev_sa] + self.ALPHA * new_q

        self.prev_sa = self.compress_sa(state, action)
        '''
        sa = self.compress_sa(state, action)
        new_q = reward + self.GAMMA * max([self.qtable[self.compress_sa(state, a)] for a in valid_actions])
        self.qtable[sa] = (1 - self.ALPHA) * self.qtable[sa] + self.ALPHA * new_q

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        print 't = ' + str(t); print self.qtable

    def compress_sa(self, state, action):
        """Given state, action pair, compress it into a smaller representation space"""
        # Recall: state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])
        wp, light, oncoming, left, right = state

        # sa = [follow waypoint, run red light, left turn yield fail, right turn yield fail, don't move]
        sa = [False for x in range(5)]

        # Follow waypoint
        if action is wp:
            sa[0] = True

        # Run red light
        if (action is 'forward' or action is 'left') and light is 'red':
            sa[1] = True

        # Left turn yield fail
        if action is 'left' and (oncoming is 'forward' or oncoming is 'right') and light is 'green':
            sa[2] = True

        # Right turn yield fail
        if action is 'right' and left is 'forward' and light is 'red':
            sa[3] = True

        # Don't move
        if action is None:
            sa[4] = True

        return tuple(sa)


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
