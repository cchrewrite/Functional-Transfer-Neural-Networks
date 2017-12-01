        
# Weight: y = W0 ^ 2 * (x + W1) ^ 2
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
        #self.W2T = self.W[2].T
        self.INP = []
        self.X = []
        self.DX = []
        self.WW = []

    def fprop(self, inputs):

        if inputs.ndim == 4:
            inputs = inputs.reshape(inputs.shape[0], -1)
  
        self.INP = inputs
        a = numpy.zeros([inputs.shape[0], self.W[1].shape[1]], dtype=numpy.float32)
        #self.X = numpy.zeros([inputs.shape[0], self.W[1].shape[1], self.W[1].shape[0]], dtype=numpy.float32)
        #self.DX = numpy.zeros([inputs.shape[0], self.W[1].shape[1], self.W[1].shape[0]], dtype=numpy.float32)
        self.W0T = self.W[0].T
        self.W1T = self.W[1].T
        #self.W2T = self.W[2].T

        self.X = self.INP[:,None] + self.W1T    
        self.DX = 2 * self.X
        self.X = self.X * self.X
        self.WW = self.W[0] * self.W[0]
        
        #a = numpy.dot(self.X, self.WW) + self.b
        a = numpy.sum(self.X * self.WW.T, axis = 2) + self.b
        
        return a

    def bprop(self, h, igrads):
             
        
        ograds = numpy.zeros([igrads.shape[0], self.W[1].shape[0]], dtype=numpy.float32)

        for i in xrange(igrads.shape[0]):
            ograds[i] = numpy.dot(igrads[i], self.DX[i] * self.WW.T)

        # The following computation consumes more time.
        #ograds = numpy.sum(self.DX.transpose(0,2,1) * (igrads[:,None] * self.WW), axis = 2)
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
     
        
        der_W0 = 2 * self.W[0]
        for i in xrange(self.INP.shape[0]):
            grad_W[0] += deltas[i] * (self.X[i].T * der_W0)
            grad_W[1] += deltas[i] * (self.DX[i] * self.WW.T).T
        
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