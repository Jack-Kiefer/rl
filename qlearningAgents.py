# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.q_vals = dict() # self.q_vals[state][action] = <q_value>

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise

          state: represents the current state
          action: represents action taken
          return: float, representing Q(state,action)
        """
        if state in self.q_vals.keys():
          if action in self.q_vals[state].keys():
            return self.q_vals[state][action]
        else:
          self.q_vals[state] = dict()
        self.q_vals[state][action] = 0.0
        return 0.0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.

          state: represents the current state
          return: float, representing V(state) = max_{a} Q(state,a)
        """
        max_q = -math.inf
        available_actions = self.getLegalActions(state)
        if len(available_actions) == 0:
          return 0.0
        for action in available_actions:
          max_q = max(max_q, self.getQValue(state, action))
        return max_q

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.

          state: represents the current state
          return: action, representing best action to take at state according to Q-values, None if not possible
        """
        available_actions = self.getLegalActions(state)
        max_actions = []
        max_qvalue = self.computeValueFromQValues(state)
        if len(available_actions) == 0:
          return None
        for action in available_actions:
          if self.getQValue(state, action) == max_qvalue:
            max_actions.append(action)
        return random.choice(max_actions)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this upon observing a
          (state => action => nextState and reward) transition.
          You should do your Q-value update here.

          NOTE: You should never call this function,
          it will be called on your behalf

          state: represents the current state
          action: represents action taken
          nextState: represents the resulting state (s')
          reward: float, represents the immediate reward gained, R(s,a,s')
          this method should update class variables, return nothing
        """
        "*** YOUR CODE HERE ***"
        original = self.getQValue(state, action)
        new = original + self.alpha * (reward + self.discount * \
          self.computeValueFromQValues(nextState) - original)
        self.q_vals[state][action] = new

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)


          state: represents the current state
          return: action, representing the action taken according to the epsilon greedy policy
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
          return None
        choose_rdm = util.flipCoin(self.epsilon)
        if not choose_rdm:
          return self.getPolicy(state)
        return random.choice(legalActions)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

class QLearningAgentCountExploration(QLearningAgent):
    def __init__(self, k=2, **args):
        self.visitCount = util.Counter() 
        self.k = k
        QLearningAgent.__init__(self, **args)
    
    def get_f_vals(self, u, n):
      return u + self.k / (n + 1.0)
      
    # Feel free to add helper functions here
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this upon observing a
          (state => action => nextState and reward) transition.
          You should do your Q-value update here.

          You should update the visit count in this function 
          for the current state action pair.

          NOTE: You should never call this function,
          it will be called on your behalf

          state: represents the current state
          action: represents action taken
          nextState: represents the resulting state (s')
          reward: float, represents the immediate reward gained, R(s,a,s')
          this method should update class variables, return nothing
        """ 
        # visit the current Q-state Q(state, action)
        self.visitCount[(state, action)] += 1
        possible_actions = self.getLegalActions(nextState)
        f_value = 0
        if len(possible_actions) > 0:
          f_value = -math.inf
          for action_p in possible_actions:
            f_val = self.get_f_vals(self.getQValue(nextState, action_p), \
              self.visitCount[(nextState, action_p)])
            if f_val > f_value:
              f_value = f_val
        old_q = self.getQValue(state, action)
        new_q = old_q + self.alpha * (reward + self.discount * f_value - old_q)
        self.q_vals[state][action] = new_q

    def getAction(self, state):
        """
          Compute the action to take in the current state. 
          Break ties randomly.

          state: represents the current state
          return: action, representing the action taken according to the visit count based exploration policy
        """
        actions = []
        best_q = -math.inf
        for naction in self.getLegalActions(state):
          new_q = self.getQValue(state, naction)
          if new_q > best_q:
            actions = [naction]
            best_q = new_q
          elif new_q == best_q:
            actions.append(naction)
        print(actions)
        if len(actions) > 0:
          return random.choice(actions)
        return None


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator

          state: represents the current state
          action: represents action taken
          return: float, representing Q_w(state,action)
        """
        features = self.featExtractor.getFeatures(state, action)
        if not self.weights:
          for feature in features:
            self.weights[feature] = 0 
        return self.getWeights() * self.featExtractor.getFeatures(state, action)

    def update(self, state, action, nextState, reward):
        """
          Should update your weights based on transition

          state: represents the current state
          action: represents action taken
          nextState: represents the resulting state (s')
          reward: float, represents the immediate reward gained, R(s,a,s')
          this method should update class variables, return nothing
        """
        prev_qval = self.getQValue(state, action)
        next_actions = self.getLegalActions(nextState)
        best_nqval = 0
        if len(next_actions) > 0:
          best_nqval = -math.inf
          for naction in next_actions:
            nqval = self.getQValue(nextState, naction)
            if nqval > best_nqval:
              best_nqval = nqval
        difference = reward + self.discount * best_nqval - prev_qval
        features = self.featExtractor.getFeatures(state, action).copy()
        for feature in features.keys():
          features[feature] *= difference * self.alpha
        self.weights += features

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print(self.weights)