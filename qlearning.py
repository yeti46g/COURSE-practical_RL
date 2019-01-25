from collections import defaultdict
import random, math
import numpy as np

class QLearningAgent :
    
    def __init__(self,alpha,epsilon,discount,get_legal_actions):
        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda :0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
    
    def get_qvalue(self,state,action):
        return self._qvalues[state][action]
    
    def set_qvalue(self,state,action,value):
        self._qvalues[state][action] = value
    
    def get_value(self,state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return 0.0
        possible_values = [self.get_qvalue(state,action) for action in possible_actions]
        state_value = np.max(possible_values)
        return state_value
    
    def update(self,state,action,reward,next_state):
        gamma = self.discount
        learning_rate = self.alpha
        qvalue = (1-learning_rate) * self.get_qvalue(state,action) + learning_rate * (reward + gamma * self.get_value(next_state))
        self.set_qvalue(state,action,qvalue)
        
    def get_best_action(self,state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None
        possible_q_values = [self.get_qvalue(state,action) for action in possible_actions]
        index = np.argmax(possible_q_values)
        best_action = possible_actions[index]
        return best_action
    
    def get_action(self,state):
        possible_actions = self.get_legal_actions(state)
        action = None
        if len(possible_actions) == 0:
            return None
        epsilon = self.epsilon
        choice = np.random.random() > epsilon
        if choice:
            chosen_action = self.get_best_action(state)
        else:
            chosen_action = random.choice(possible_actions)
        return chosen_action
