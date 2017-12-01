class Functional(Layer):

    def __init__(self, idim, odim,
                 rng=None,
                 irange=0.1):

        super(Functional, self).__init__(rng=rng)

        self.idim = idim
        self.odim = odim

        self.W = self.rng.uniform(
            -irange, irange,
            (self.idim, self.odim))
        
        self.b = numpy.zeros((self.odim,), dtype=numpy.float32)
            

    def fprop(self, inputs):

        if inputs.ndim == 4:
            inputs = inputs.reshape(inputs.shape[0], -1)

        # Weight: y = W0 * W0 * x ; Bias: z = y + B0  
        a = numpy.dot(inputs, self.W * self.W) + self.b
       
        return a

    def bprop(self, h, igrads):
        
        # Weight: y = W0 * W0 * x ; Bias: z = y + B0
        ograds = numpy.dot(igrads, (self.W * self.W).T)
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
        
        # Weight: y = W0 * W0 * x ; Bias: z = y + B0
        der_W = 2 * self.W
        grad_W = numpy.dot(inputs.T, deltas) * der_W
        grad_b = numpy.sum(deltas, axis=0)

        grad_W = numpy.clip(grad_W, -1000.0, +1000.0)
        grad_b = numpy.clip(grad_b, -1000.0, +1000.0)
        
        return [grad_W, grad_b]

    def get_params(self):
        return [self.W, self.b]

    def set_params(self, params):
        self.W = params[0]
        self.b = params[1]

    def get_name(self):
        return 'functional'