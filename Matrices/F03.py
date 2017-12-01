# Weight: y = W0 * x * x + W1 * x 
class Functional(Layer):

    def __init__(self, idim, odim,
                 rng=None,
                 irange=0.1):

        super(Functional, self).__init__(rng=rng)

        self.idim = idim
        self.odim = odim

        self.W = self.rng.uniform(
            -irange, irange,
            (2, self.idim, self.odim))
        
        self.b = numpy.zeros((self.odim,), dtype=numpy.float32)

        self.RM = []
        self.W0T = self.W[0].T
        self.W1T = self.W[1].T
        self.INP = []
        self.INP2 = []

    def fprop(self, inputs):

        if inputs.ndim == 4:
            inputs = inputs.reshape(inputs.shape[0], -1)

        self.INP = inputs
        self.INP2 = inputs * inputs
        a = numpy.dot(self.INP2, self.W[0]) + numpy.dot(self.INP, self.W[1]) + self.b      

        return a

    def bprop(self, h, igrads):

        ograds = 2 * self.INP * numpy.dot(igrads, self.W[0].T) + numpy.dot(igrads, self.W[1].T)

        return igrads, ograds

    def bprop_cost(self, h, igrads, cost):

        if cost is None or cost.get_name() == 'mse':
            return self.bprop(h, igrads)
        else:
            raise NotImplementedError('Functional.bprop_cost method not implemented '
                                      'for the %s cost' % cost.get_name())

    def pgrads(self, inputs, deltas, l1_weight=0, l2_weight=0):

        if inputs.ndim == 4:
            inputs = inputs.reshape(inputs.shape[0], -1)

        grad_W = numpy.zeros(self.W.size, dtype=numpy.float32).reshape(self.W.shape)
        
        grad_W[0] = numpy.dot(self.INP2.T, deltas)
        grad_W[1] = numpy.dot(self.INP.T, deltas)
        grad_b = numpy.sum(deltas, axis=0)
      
        grad_W = numpy.clip(grad_W, -1000.0, 1000.0)
        grad_b = numpy.clip(grad_b, -1000.0, 1000.0)

  
        return [grad_W, grad_b]

    def get_params(self):
        return [self.W, self.b]

    def set_params(self, params):
        self.W = params[0]
        self.b = params[1]

    def get_name(self):
        return 'functional'