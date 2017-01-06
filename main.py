

import math
import uuid
from random import choice, random, uniform


def weighted_choice(choices):
    assert bool(choices)
    total = sum(w for c, w in choices.items())
    r = uniform(0, total)
    upto = 0
    for c, w in choices.items():
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"


def curved_randint(start, end):
    choices = {}
    r = list(range(start, end + 1))

    for i in range(0, (end + 1 - start) // 2):
        val = i + 1
        choices[r[i]] = val
        choices[r[-i - 1]] = val
    return weighted_choice(choices)


class Network:
    def __init__(
            self, inputs, outputs, nodes, weights,
            mutation_chance=.05, learning_rate=1.):
        self.mutation_chance = mutation_chance
        self.learning_rate = learning_rate
        self.inputs = {input_.uid: input_ for input_ in inputs}
        self.outputs = {output.uid: output for output in outputs}
        self.nodes = {node.uid: node for node in nodes}
        self.weights = {weight.uid: weight for weight in weights}

    def raw(self):
        result = {
            'inputs': {},
            'outputs': {},
            'nodes': {},
            'weights': {},
        }
        for key, input_ in self.inputs.items():
            result['inputs'][key] = {
                'bias': input_.bias,
            }
        for key, output in self.outputs.items():
            result['outputs'][key] = {
                'bias': output.bias,
            }
        for key, node in self.nodes.items():
            result['nodes'][key] = {
                'bias': node.bias,
            }
        for key, weight in self.weights.items():
            result['weights'][key] = {
                'amount': weight.amount,
                'input_': weight.input_,
                'output': weight.output,
            }
        return result

    @classmethod
    def load(cls, dump):
        """Load network from dump"""
        inputs = {}
        outputs = {}
        nodes = {}
        weights = []

        for input_ in dump['inputs']:
            key = input_['uid']
            inputs[key] = (Unit(**input_))
        for output in dump['outputs']:
            key = output['uid']
            outputs[key] = (Unit(**output))
        for node in dump['nodes']:
            key = node['uid']
            nodes[key] = (Unit(**node))
        for weight in dump['weights']:
            key = weight['uid']
            weight['input_'] = inputs[weight['input_']]
            weight['output'] = output[weight['output']]
            weights.append(Weight(**weight))

        return cls(
            list(inputs.values()),
            list(outputs.values()),
            list(nodes.values()),
            weights,
            dump['mutation_chance'],
        )

    @classmethod
    def evolve(cls, *parents):
        """Evolve from N parents"""
        parents_amount = len(parents)
        pool = {
            'inputs': {},
            'outputs': {},
            'nodes': {},
            'weights': {},
            'mutation_chances': [],
        }
        genes = {
            'inputs': {},
            'outputs': {},
            'nodes': {},
            'weights': {},
        }

        for parent in parents:
            parent = parent.raw()
            pool['mutation_chances'].append(parent['mutation_chance'])
            for key, input_ in parent['inputs'].items():
                try:
                    pool['inputs'][input_[key]]['bias'].append(input_['bias'])
                except KeyError:
                    pool['inputs'][input_[key]] = {'bias': [input_['bias']]}
            for key, output in parent['outputs'].items():
                try:
                    pool['outputs'][output[key]]['bias'].append(output['bias'])
                except KeyError:
                    pool['outputs'][output[key]] = {'bias': [output['bias']]}
            for key, node in parent['nodes'].items():
                try:
                    pool['nodes'][node[key]]['bias'].append(node['bias'])
                except KeyError:
                    pool['nodes'][node[key]] = {'bias': [node['bias']]}
            for key, weight in parent['weights'].items():
                try:
                    pool['weights'][weight[key]]['amount'].append(
                        weight['amount'])
                    pool['weights'][weight[key]]['input_'].append(
                        weight['input_'])
                    pool['weights'][weight[key]]['output'].append(
                        weight['output'])
                except KeyError:
                    pool['weights'][weight[key]] = {
                        'amount': [weight['amount']],
                        'input_': [weight['input_']],
                        'output': [weight['output']],
                    }

        mc = choice(pool['mutation_chances'])
        mc = uniform(-10., 10.) if random < mc else mc
        genes['mutation_chance'] = mc
        for type_ in ['inputs', 'outputs', 'nodes']:
            for key, nodes in pool[type_].items():
                nodes = nodes + [
                    None for i in range(parents_amount - len(nodes))
                ]
                node = choice(nodes)
                if node and (type_ != 'nodes' or random() > mc):
                    # should only not fire if there is no node or
                    # the type_ is nodes and it is below the threshold
                    # -> random chance of node dying
                    bias = (
                        round(uniform(-10., 10.), 2)
                        if random() < mc  # random chance of mutation
                        else node['bias']
                    )
                    genes[type_].append({
                        'bias': bias,
                        'uid': key,
                    })
                if random() < mc:  # random chance of gaining new node
                    bias = round(uniform(-10., 10.), 2)
                    genes[type_].append({
                        'bias': bias,
                        'uid': None,
                    })
        for key, weights in pool['weights'].items():
            weights = weights + [
                None for i in range(parents_amount - len(weights))
            ]
            weight = choice(weights)
            if weight and random() > mc:  # random chance of weight dying
                amount = weight['amount']
                variation = curved_randint(-1000, 1000) / 100  # mutation
                amount += min(max(variation, -10), 10)
                genes['weights'].append({
                    'uid': key,
                    'amount': amount,
                    'input_': weight['input_'],
                    'output': weight['output'],
                })
            if random() < mc:  # random chance of gaining new weight
                amount = round(uniform(-10., 10.), 2)
                genes['weights'].append({
                    'uid': None,
                    'amount': amount,
                    'input_': choice(genes['inputs'] + genes['nodes'])['uid'],
                    'output': choice(genes['outputs'] + genes['nodes'])['uid'],
                })
        return cls.load(genes)

    # def backpropagate(self, expected):
    #     def error_function(result, expected):
    #         return abs(result - expected) ** 2

    #     def activation_function(value):
    #         return math.tanh(value)

    #     L = self.learning_rate
    #     pairs = zip(self.output(), expected)
    #     for res, exp in pairs:
    #         E = error_function(res, exp)  # Error
    #         for weight in self.weights:

    #     raise NotImplementedError

    def output(self):
        return {key: out.value for key, out in self.outputs.items()}

    def trigger(self, values):
        for key, value in values.items():
            self.inputs[key].trigger(value)
        return self.output()


class Weight:
    def __init__(self, amount, input_, output, uid=None):
        self.uid = uid if uid else uuid.uuid4()
        self.amount = amount
        self.input_ = input_
        self.output = output
        input_.outputs.add(self)
        output.inputs.add(self)

    def trigger(self, value):
        self.output.trigger(value * self.amount)


class Unit:
    def __init__(self, bias=-.5, trigger_threshold=1, uid=None):
        self.active = False
        self.uid = uid if uid else uuid.uuid4()
        self.bias = bias
        self.inputs = set()
        self.outputs = set()
        self.value = 0
        self.times_triggered = 0
        self.trigger_threshold = trigger_threshold

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def trigger(self, value):
        if not self.times_triggered:
            self.value = 0
            self.active = True
        self.value += value
        if self.times_triggered >= self.trigger_threshold:
            self.times_triggered = 0
            self.value = self.sigmoid(self.value + self.bias)
            for output in self.outputs:
                output.trigger(self.value)
        self.times_triggered += 1


if __name__ == "__main__":
    dump = {
        'inputs': [
            {
                'uid': 'in1',
            },
            {
                'uid': 'in2',
            },
            {
                'uid': 'in3',
            },
        ],
        'outputs': [
            {
                'uid': 'out1'
            },
            {
                'uid': 'out2'
            },
            {
                'uid': 'out3'
            },
        ],
        'nodes': [],
        'weights': [],
    }
    network = Network.load(dump)
    values = {
        'in1': 0,
        'in2': 0,
        'in3': 0,
    }
    print(network.trigger(values))
