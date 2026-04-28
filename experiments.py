from qnca import qnca
from qnca import ca_patterns
from qnca.optimizers import base, cobyla, cma, ga

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

dispositivo = 'GPU' if torch.cuda.is_available() else 'CPU'

DIRETORIO_PADRAO = 'D:\\Dropbox\\Projetos\\pessoal\\QNCA\\results\\'

experiments = base.QNCAGlobalOptimizer(ca_patterns.rules, cobyla.QNCAOptimizerCOBYLA, path = DIRETORIO_PADRAO)
experiments.global_training()
experiments.fine_tunning()

experiments = base.QNCAGlobalOptimizer(ca_patterns.rules, cma.QNCAOptimizerCMA, path = DIRETORIO_PADRAO)
experiments.global_training()
experiments.fine_tunning()

experiments = base.QNCAGlobalOptimizer(ca_patterns.rules, ga.QNCAOptimizerGA, path = DIRETORIO_PADRAO, shots = 100)
experiments.global_training()
experiments.fine_tunning(k = 30)




#print(experiments.k_perturbed_best('28','30', k= 2))


