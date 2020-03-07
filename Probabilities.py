# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:43:55 2020

@author: Cristi
"""


# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from pomegranate import DiscreteDistribution
from pomegranate import ConditionalProbabilityTable
from pomegranate import Node
from pomegranate import BayesianNetwork

chinese = DiscreteDistribution({True: 0.04, False: 0.96})
overThirty = DiscreteDistribution({True: 0.6, False: 0.4})
playPingPong = ConditionalProbabilityTable(
        [[True,  True,  True,  0.15],
         [True,  True,  False, 0.85],
         [True,  False, True,  0.3],
         [True,  False, False, 0.7],
         [False, True,  True,  0.05],
         [False, True,  False, 0.95],
         [False, False, True,  0.08],
         [False, False, False, 0.92]], [chinese, overThirty])


s1 = Node(chinese, name="chinese")
s2 = Node(overThirty, name="overThirty")
s3 = Node(playPingPong, name="playPingPong")


model = BayesianNetwork("Ping pong problem")
model.add_states(s1, s2, s3)
model.add_edge(s1, s3)
model.add_edge(s2, s3)
model.bake()


print('Case 2:', model.probability([True, True, False]))
print('Case 3:', model.probability([True, False, True]))
print('Case 8:', model.probability([False, False, False]))

print('Predict without further infos')
print(model.predict_proba({}))

print('Predict if chinese')
print(model.predict_proba({'chinese': True}))


print('Predict if chinese and plays ping pong')
print(model.predict_proba({'chinese': True, 'playPingPong': True}))