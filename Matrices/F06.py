# Weight: y = Relu(W) * Sgn(B) * x        
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
        
        self.W = self.rng.uniform(
            0, 2 * irange,
            (self.idim, self.odim))
        self.B_Relu = self.rng.uniform(
            -irange, irange,
            (self.idim, self.odim))
        self.B_Relu = ((self.B_Relu >= 0) + 0) - ((self.B_Relu < 0) + 0)
        
        self.b = numpy.zeros((self.odim,), dtype=numpy.float32)

    def fprop(self, inputs):

        if inputs.ndim == 4:
            inputs = inputs.reshape(inputs.shape[0], -1)

        a = numpy.dot(inputs, numpy.clip(self.W, 0, 1000) * self.B_Relu) + self.b
        
        return a

    def bprop(self, h, igrads):

        ograds = numpy.dot(igrads, (numpy.clip(self.W, 0, 1000) * self.B_Relu).T)
        
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

        # B-Relu
        der_W = (self.W >= 0)
        grad_W = numpy.dot(inputs.T, deltas) * der_W * self.B_Relu 
  
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