# Weight: y = tanh(W0 * x + W1 * yc + W2)
class Functional(Layer):

    def __init__(self, idim, odim,
                 rng=None,
                 irange=0.1):

        super(Functional, self).__init__(rng=rng)

        self.idim = idim
        self.odim = odim

        self.W = self.rng.uniform(
            -irange, irange,
            (3, self.idim, self.odim))
            
        self.X = self.rng.uniform(
            -irange, irange,
            (self.idim, self.odim)).T * 0
            

        
        self.b = numpy.zeros((self.odim,), dtype=numpy.float32)
            
            
        self.RM = []
        self.W0T = self.W[0].T
        self.W1T = self.W[1].T
        self.W2T = self.W[2].T
        self.INP = []
        self.C = []
        self.DX = []

    def fprop(self, inputs):

        if inputs.ndim == 4:
            inputs = inputs.reshape(inputs.shape[0], -1)
  
        self.INP = inputs.reshape(self.idim)
        
        self.W0T = self.W[0].T
        self.W1T = self.W[1].T
        self.W2T = self.W[2].T
        
        self.C = self.X * 1.0

        self.X = numpy.tanh(self.INP * self.W0T + self.C * self.W1T + self.W2T)
        
        self.DX = 1 - self.X * self.X

        a = (numpy.sum(self.X, axis = 1) + self.b).reshape(1,self.odim)
        
        return a

    def bprop(self, h, igrads):
             
        ograds = numpy.dot(igrads, self.W0T * self.DX)

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

        grad_W[0] = numpy.dot(inputs.T, deltas) * self.DX.T
        grad_W[1] = deltas[0] * self.C.T * self.DX.T
        grad_W[2] = deltas[0] * (self.W2T * 0 + 1).T * self.DX.T
     
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