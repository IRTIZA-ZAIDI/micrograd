# micrograd

A minimal scalar-valued autograd engine and a tiny neural network library built on top of it.

This repository contains:

* `micrograd.engine.Value`: a scalar that records a computation graph and supports reverse-mode automatic differentiation (backprop).
* `micrograd.nn`: a small neural network module system (Neuron / Layer / MLP) using `Value` parameters.

The code is intentionally small and readable so you can learn how backprop and basic MLPs work under the hood.

---

## Contents

* **Autograd core**

  * Scalar `Value(data, grad)` nodes
  * Operator overloading builds the computation graph
  * `backward()` performs a topological traversal and applies chain rule

* **Neural nets**

  * `Neuron(nin, nonlin=True)` with learnable weights and bias
  * `Layer(nin, nout)` as a list of neurons
  * `MLP(nin, [h1, h2, ..., out])` stacked layers
  * `Module.zero_grad()` and `Module.parameters()` utilities

* **Learning notebook**

  * `learning/micrograd.ipynb` for step-by-step exploration

* **Tests**

  * `tests/test.py` compares gradients against PyTorch for sanity

---

## Project structure

```
micrograd/
  __init__.py
  engine.py     # Value class + autograd
  nn.py         # Module, Neuron, Layer, MLP
learning/
  micrograd.ipynb
tests/
  test.py
```

---

## Requirements

* Python 3.9+ (recommended: 3.10+)

Optional (for running tests):

* `torch` (tests compare gradients to PyTorch)

Optional (for notebooks):

* `jupyter` or `ipykernel`

---

## Installation

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows PowerShell

pip install --upgrade pip
```

If you want to run tests:

```bash
pip install pytest torch
```

If you want to run the notebook:

```bash
pip install jupyter
```

Note: This repo is a simple Python package layout. If you run scripts from the repo root, `import micrograd` will work without additional installation.

---

## Quickstart: autograd with `Value`

```python
from micrograd.engine import Value

x = Value(-4.0)
z = 2 * x + 2 + x
q = z.relu() + z * x
h = (z * z).relu()
y = h + q + q * x

y.backward()

print("y =", y.data)
print("dy/dx =", x.grad)
```

### Supported ops in this implementation

`Value` supports:

* `+`, `-`, unary negation
* `*`, `/`
* power: `x ** k` for `k` as `int` or `float`
* `relu()`
* `backward()` to compute gradients

---

## Quickstart: a tiny MLP

```python
from micrograd.nn import MLP

# 3 inputs -> [4 hidden, 4 hidden, 1 output]
model = MLP(3, [4, 4, 1])

x = [1.0, -2.0, 0.5]
out = model(x)          # output is a Value
print(out.data)

# Backprop through the whole network
model.zero_grad()
out.backward()

# Access parameters and gradients
params = model.parameters()
print(len(params))
print(params[0].data, params[0].grad)
```

---

## Example: simple training loop (binary classification style)

This is a minimal pattern you can reuse. Here we treat targets as `-1` or `+1` and optimize a squared loss.

```python
from micrograd.nn import MLP

# toy dataset (3 features -> binary target)
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])
lr = 0.05

for step in range(200):
    # forward
    ypred = [model(x) for x in xs]

    # squared loss
    loss = sum((yout - ygt) ** 2 for yout, ygt in zip(ypred, ys))

    # backward
    model.zero_grad()
    loss.backward()

    # SGD update
    for p in model.parameters():
        p.data += -lr * p.grad

    if step % 20 == 0:
        print(step, loss.data)
```

---

## Running tests

From the repo root:

```bash
pytest -q
```

`tests/test.py` checks:

* forward values match PyTorch
* gradients match PyTorch for a set of composed operations

If you see an import error for `torch`, install it:

```bash
pip install torch
```

---

## Notebook

Open and run:

* `learning/micrograd.ipynb`

This is the best place to explore the computation graph, gradients, and how the MLP is built from scalar operations.

---

## Notes

* This is a learning-focused implementation: scalar-only, no tensors, and no GPU support.
* The design mirrors the core ideas of reverse-mode autodiff used in modern deep learning frameworks, but stripped down to the essentials.

---

## Attribution

This style of implementation is widely known as “micrograd” and is commonly associated with educational materials popularizing minimal autograd engines.
