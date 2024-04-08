import random
from toynn.nn.engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
    def parameters(self):  m
        return []

class Neuron(Module):
    def __init__(self, nin):
        # Assiging a weight to every input, randomly drawn from a uniform distribution.
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        assert len(x) == len(self.w)
        acts = sum([wi * xi for wi, xi in zip(self.w, x)], self.b)
        out = acts.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):
    def __init__(self, nin, nouts):
        stacks = [nin] + nouts
        self.layers = [Layer(stacks[i], stacks[i+1]) for i in range(len(stacks) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
