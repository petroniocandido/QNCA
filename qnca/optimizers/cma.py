import cma
import numpy as np

class QNCAOptimizerCMA(QNCAOptimizer):
  name = 'CMA'
  def __init__(self, **kwargs):
    super(QNCAOptimizerCMA, self).__init__(**kwargs)
    self.maxiter = kwargs.get('maxiter', 60)
    self.popsize = kwargs.get('population', 15)

  def training_loop(self):
    parametros = np.random.randn(self.num_param) * (2 * np.pi)

    objetivo = lambda x: self.funcao_custo(x)

    options = {'maxiter': self.maxiter, 'popsize': self.popsize}

    sigma0 = .5        # Desvio padrão inicial (tamanho do passo de busca)

    es = cma.fmin2(objetivo, parametros, sigma0, options)

    return es
