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

likeCompetitions = DiscreteDistribution({True: 0.35, False: 0.65})
isRich = DiscreteDistribution({True: 0.1, False: 0.9})
overThirty = DiscreteDistribution({True: 0.6, False: 0.4})
makeHighPerformanceSport = ConditionalProbabilityTable(
        [[True,   True,  True,  True, 0.27],
         [True,   True,  True,  False, 0.73],
         [True,   True,  False, True, 0.37],
         [True,   True,  False, False, 0.63],
         [True,   False,  True,  True, 0.15],
         [True,   False,  True,  False, 0.85],
         [True,   False,  False, True, 0.19],
         [True,   False,  False, False, 0.81],
         [False,   True,  True,  True, 0.08],
         [False,   True,  True,  False, 0.92],
         [False,   True,  False, True, 0.10],
         [False,   True,  False, False, 0.90],
         [False,   False,  True,  True, 0.03],
         [False,   False,  True,  False, 0.97],
         [False,   False,  False, True, 0.06],
         [False,   False,  False, False, 0.94]], [likeCompetitions, isRich, overThirty])

isChinese = DiscreteDistribution({True: 0.18, False: 0.82})
preferIndoorSports = DiscreteDistribution({True: 0.35, False: 0.65})

participateAtPingPongCompetitions = ConditionalProbabilityTable(
        [[True,   True,  True,  True, 0.45],
         [True,   True,  True,  False, 0.55],
         [True,   True,  False, True, 0.05],
         [True,   True,  False, False, 0.95],
         [True,   False,  True,  True, 0.12],
         [True,   False,  True,  False, 0.88],
         [True,   False,  False, True, 0.03],
         [True,   False,  False, False, 0.97],
         [False,   True,  True,  True, 0.35],
         [False,   True,  True,  False, 0.65],
         [False,   True,  False, True, 0.04],
         [False,   True,  False, False, 0.96],
         [False,   False,  True,  True, 0.07],
         [False,   False,  True,  False, 0.93],
         [False,   False,  False, True, 0.01],
         [False,   False,  False, False, 0.99]], [isChinese, makeHighPerformanceSport, preferIndoorSports])

participateAtTennisCompetitions = ConditionalProbabilityTable(
        [[True,   True,  True,  True, 0.1],
         [True,   True,  True,  False, 0.9],
         [True,   True,  False, True, 0.24],
         [True,   True,  False, False, 0.76],
         [True,   False,  True,  True, 0.02],
         [True,   False,  True,  False, 0.98],
         [True,   False,  False, True, 0.07],
         [True,   False,  False, False, 0.93],
         [False,   True,  True,  True, 0.15],
         [False,   True,  True,  False, 0.85],
         [False,   True,  False, True, 0.33],
         [False,   True,  False, False, 0.67],
         [False,   False,  True,  True, 0.09],
         [False,   False,  True,  False, 0.91],
         [False,   False,  False, True, 0.04],
         [False,   False,  False, False, 0.96]], [isChinese, makeHighPerformanceSport, preferIndoorSports])

s1 = Node(likeCompetitions, name="likeCompetitions")
s2 = Node(isRich, name="isRich")
s3 = Node(overThirty, name="overThirty")
s4 = Node(isChinese, name="isChinese")
s5 = Node(makeHighPerformanceSport, name="makeHighPerformanceSport")
s6 = Node(preferIndoorSports, name="preferIndoorSports")
s7 = Node(participateAtPingPongCompetitions, name="participateAtPingPongCompetitions")
s8 = Node(participateAtTennisCompetitions, name="ParticipateAtTennisCompetitions")

model = BayesianNetwork("Choose sport problem")

model.add_states(s1, s2, s3, s4, s5, s6, s7, s8)
model.add_edge(s1, s5)
model.add_edge(s2, s5)
model.add_edge(s3, s5)
model.add_edge(s4, s7)
model.add_edge(s5, s7)
model.add_edge(s6, s7)
model.add_edge(s4, s8)
model.add_edge(s5, s8)
model.add_edge(s6, s8)

model.bake()


print(model.probability([False, False, False, True, True, False, False, False]))
print(model.probability([True, True, False, True, True, False, False, True]))

print('Predict without further infos')
print(model.predict_proba({}))

print('Predict if likes competitions')
print(model.predict_proba({'likeCompetitions': True}))

print('Predict if prefer indoor competitions')
print(model.predict_proba({'preferIndoorSports': True}))

print('Predict if rich and chinese')
print(model.predict_proba({'isRich': True, 'isChinese': True}))
