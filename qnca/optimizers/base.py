from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import  Aer, AerSimulator
from qiskit import transpile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import json
import os
from .qnca import QNCA

class QNCAOptimizer(object):
  def __init__(self, **kwargs):
    self.name = None
    self.min_loss = np.inf
    self.best_param = None
    self.loss_history = []
    self.param_history = []
    self.kwargs = kwargs
    self.pattern = kwargs['pattern']
    T,n = self.pattern.shape
    self.T = T
    self.n = n
    self.initial = self.pattern[0,:].tolist()
    self.dispositivo = 'GPU' if torch.cuda.is_available() else 'CPU'
    self.operator = kwargs['operator']
    self.num_param = int(str(self.operator)[:-1])
    self.shots = kwargs.get('shots',33)

    if 'backend' in kwargs:
      self.backend = kwargs['backend']
    else:
      self.backend = Aer.get_backend('qasm_simulator', device=self.dispositivo)

  def clean(self):
    self.loss_history = []
    self.param_history = []
    self.min_loss = np.inf
    self.best_param = None

  def log(self, loss, param):
    self.loss_history.append(loss)
    self.param_history.append(param.tolist())
    if loss < self.min_loss:
      self.min_loss = loss
      self.best_param = param.tolist()
      print("NEW MIN: {}\t{}".format(self.min_loss, self.best_param))

  def funcao_custo(self, parametros):
    evolution = np.zeros((self.T,self.n))

    mse = 0.0

    for t in range(self.T-1):
      qc = QNCA(operator = self.operator, initial=self.initial, T = t+1, backend = self.backend, parametros = parametros)

      job = self.backend.run(qc.final_circuit, shots=self.shots)
      counts = job.result().get_counts()

      statistics = {k: {d : 0 for d in ['0','1']} for k in range(self.n)}
      for output, count in counts.items():
        for ix, d in enumerate(reversed(output)):
          statistics[ix][d] += count

      for i in range(self.n):
        out_t_i = np.sum([int(k) * v for k,v in statistics[i].items()])/self.shots
        mse += (self.pattern[t+1,i] - out_t_i)**2

    self.log(mse, parametros)

    return mse


  def plot(self):
    plt.plot(self.loss_history)




