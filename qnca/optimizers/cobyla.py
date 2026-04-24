from scipy.optimize import minimize
from .base import QNCAOptimizer

class QNCAOptimizerCOBYLA(QNCAOptimizer):
  name = 'COBYLA'
  def __init__(self, **kwargs):
    super(QNCAOptimizerCOBYLA, self).__init__(**kwargs)

  def training_loop(self):
    parametros = np.random.rand(self.num_param) * (2 * np.pi)

    objetivo = lambda x: self.funcao_custo(x)

    # Run classical optimization (e.g., COBYLA or SLSQP)
    result = minimize(objetivo, parametros, method='COBYLA')

    return result
