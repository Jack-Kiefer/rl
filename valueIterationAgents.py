# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
        Runs self.iterations iterations of value iteration
        updates self.values, does not return anything
        """
        # Write value iteration code here
        for i in range(self.iterations):
            values = dict.copy(self.values)
            for state in self.mdp.getStates():
                best = -100000000000
                if len(self.mdp.getPossibleActions(state)) == 0: best = 0
                for action in self.mdp.getPossibleActions(state):
                    s = self.computeQValueFromValues(state, action)
                    best = max(best, s)
                values[state] = best
            self.values = values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
          state: a state in the mdp
          action: the action we took at that state
          return: float representing Q(state,action)
        """
        "*** YOUR CODE HERE ***"
        summ = 0 
        for (endstate, T) in self.mdp.getTransitionStatesAndProbs(state, action):
            R = self.mdp.getReward(state, action, endstate)
            V = self.values[endstate]
            summ += T * (R + (self.discount*V))
        return summ


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.

          state: state in mdp
          return: action type, represents best action from state (None if state is terminal)
        """
        "*** YOUR CODE HERE ***"
        best = -100000000000
        bestaction = None
        for action in self.mdp.getPossibleActions(state):
            s = self.computeQValueFromValues(state, action)
            if s > best: bestaction = action
            best = max(best, s)
        return bestaction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """
        Runs self.iterations iterations of async value iteration, only updating one state in each iteration
        updates self.values, does not return anything
        """
        for i in range(self.iterations):
            states = self.mdp.getStates()
            state = states[i%len(states)]
            best = -100000000000
            if len(self.mdp.getPossibleActions(state)) == 0: best = 0
            for action in self.mdp.getPossibleActions(state):
                s = self.computeQValueFromValues(state, action)
                best = max(best, s)
            self.values[state] = best

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """
        Runs self.iterations iterations of prioritized sweeping value iteration
        updates self.values, does not return anything
        """
        "*** YOUR CODE HERE ***"
        def findBestQValue(state):
            best = -100000000000
            if len(self.mdp.getPossibleActions(state)) == 0: best = 0
            for action in self.mdp.getPossibleActions(state):
                qval = self.computeQValueFromValues(state, action)
                best = max(best, qval)
            return best

        preddict = dict()
        for state in self.mdp.getStates():
            predecessors = set()
            for startstate in self.mdp.getStates():
                for action in self.mdp.getPossibleActions(startstate):
                    for (endstate, T) in self.mdp.getTransitionStatesAndProbs(startstate, action):
                        if state == endstate:
                            predecessors.add(startstate)
            preddict[state] = predecessors

        q = util.PriorityQueue()

        for s in self.mdp.getStates():
            diff = abs(self.values[s] - findBestQValue(s))

            q.push(s, -diff)
            
        for i in range(self.iterations):
            if q.isEmpty(): break
            s = q.pop()

            self.values[s] = findBestQValue(s)

            for p in preddict[s]:
                diff = abs(self.values[p] - findBestQValue(p))
                if diff > self.theta:
                    q.push(p, -diff)



                


