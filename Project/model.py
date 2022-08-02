import os, random
import matplotlib.pyplot as plt
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import numpy as np

%matplotlib inline

r_seed = 101

# for reproducibility
# must run this before create the BN! 
# Even in the cases where were changed the states names
gum.initRandom(r_seed) 
random.seed(r_seed)

##defining the network

bn1 = gum.fastBN('Season->SARS<-Flight;' + 'SARS->Fever;' + 'SARS->Cough;')
bn1

## cpt for Season [0 = L, 1 = H]
bn1.cpt('Season')[:] = [0.90, 0.10] 
## cpt for Flight [0 = T, 1 = F]
bn1.cpt('Flight')[:] = [0.30, 0.70] 
## cpt for SARS [0 = T, 1 = F]
bn1.cpt('SARS')[{'Season': 1, 'Flight': 0}] = [0.05, 0.95]
bn1.cpt('SARS')[{'Season': 1, 'Flight': 1}] = [0.02, 0.98]
bn1.cpt('SARS')[{'Season': 0, 'Flight': 0}] = [0.03, 0.97]
bn1.cpt('SARS')[{'Season': 0, 'Flight': 1}] = [0.02, 0.98]
## cpt for Fever [0 = T, 1 = F]
bn1.cpt('Fever')[{'SARS': 0}] = [0.90, 0.10]
bn1.cpt('Fever')[{'SARS': 1}] = [0.20, 0.80]
## cpt for Cough [0 = T, 1 = F]
bn1.cpt('Cough')[{'SARS': 0}] = [0.65, 0.35]
bn1.cpt('Cough')[{'SARS': 1}] = [0.30, 0.70]

gnb.showInference(bn1)

bn1.cpt('Season')
bn1.cpt('Flight')
bn1.cpt('SARS')
bn1.cpt('Fever')
bn1.cpt('Cough')

################################ Scenario 1 #################################
## 0 indicates that that it is true
evidence = {'Fever': 0, 'Cough': 0}
gnb.showInference(bn1, evs = evidence, engine=gum.LazyPropagation(bn1))
gnb.showPosterior(bn1, evidence, 'SARS')# calculate the updated value of prob. for SARS


################################ Scenario 2 #################################
## true, true, high, high
evidence = {'Fever': 0, 'Cough': 0, 'Season': 1, 'Flight': 0}
gnb.showInference(bn1, evs = evidence, engine=gum.LazyPropagation(bn1))
gnb.showPosterior(bn1, evidence, 'SARS')# calculate the updated value of prob. for SARS

################################ Redefine CPT for Cough #################################

bn2 = gum.fastBN('Season->SARS<-Flight;' + 'SARS->Fever;' + 'SARS->Cough[3];')
bn2

## cpt for Season [0 = L, 1 = H]
bn2.cpt('Season')[:] = [0.90, 0.10] 
## cpt for Flight [0 = T, 1 = F]
bn2.cpt('Flight')[:] = [0.30, 0.70] 
## cpt for SARS [0 = T, 1 = F]
bn2.cpt('SARS')[{'Season': 1, 'Flight': 0}] = [0.05, 0.95]
bn2.cpt('SARS')[{'Season': 1, 'Flight': 1}] = [0.02, 0.98]
bn2.cpt('SARS')[{'Season': 0, 'Flight': 0}] = [0.03, 0.97]
bn2.cpt('SARS')[{'Season': 0, 'Flight': 1}] = [0.02, 0.98]
## cpt for Fever [0 = T, 1 = F]
bn2.cpt('Fever')[{'SARS': 0}] = [0.90, 0.10]
bn2.cpt('Fever')[{'SARS': 1}] = [0.20, 0.80]
## cpt for Cough [0 = Mi, 1 = Mo, 2 = Se]
bn2.cpt('Cough')[{'SARS': 0}] = [0.10, 0.20, 0.70]
bn2.cpt('Cough')[{'SARS': 1}] = [0.75, 0.15, 0.10]

gnb.showInference(bn2)

bn2.cpt('Season')
bn2.cpt('Flight')
bn2.cpt('SARS')
bn2.cpt('Fever')
bn2.cpt('Cough')

################################ Scenario 3 #################################
## true, true, high, high
evidence2 = {'Fever': 0, 'Cough': 2, 'Season': 0, 'Flight': 1}
gnb.showInference(bn2, evs = evidence2, engine=gum.LazyPropagation(bn2))
gnb.showPosterior(bn2, evidence2, 'SARS')# calculate the updated value of prob. for SARS

################################ Scenario 4 #################################
## true, true, high, high
evidence3 = {'SARS': 0, 'Cough': 2, 'Fever': 0}
gnb.showInference(bn2, evs = evidence3, engine=gum.LazyPropagation(bn2))
gnb.showPosterior(bn2, evidence3, 'Flight')