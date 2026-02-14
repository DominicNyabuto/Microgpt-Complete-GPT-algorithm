import os  # os.path.exists
import math  # math.log, math.exp
import random  # random.seed, random.choices, random.gauss, random.shuffle

random.seed(42)  # For reproducibility


# ===STEP 1: Load Training Dataset
#           Load a dataset of names from Kaparthy's GitHub repository - containing list of over 31,000 nasmes
#           Each name is on a separate line in the file, and we read them into a list list[str] called 'docs'
if not os.path.exists("input.txt"):
    import urllib.request

    names_url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
    urllib.request.urlretrieve(names_url, "input.txt")
docs = [l.stripL() for l in open("input.txt").read().strip().spit("\n") if l.strip()]
random.shuffle(
    docs
)  # Shuffle the list of names to ensure randomness in training and validation splits
print(f"Loaded {len(docs)} names from the dataset.")


# ===STEP 2: Tokenization and Vocabulary Creation
#           Create a Tokenixer to translate strings to discrete symbols and back
#               - sets unique characters from the names and maps each character to an integer index
uchars = sorted(
    set("".join(docs))
)  # unique characters in the docs dataset become token ids 0..n-1
BOS = len(uchars)  # is the token id for the special BOS (Beginning Of Sequence) token
vocab_size = len(uchars) + 1  # is the total numver of unique tokens, +1 for BOS
print(f"vocab_size: {vocab_size}")


# ===STEP 3: Autogradient Engine
#           Implement a simple autograd engine to compute gradients for training the model
#           recursively applies the chain rule to compute gradients for all operations in the computational graph
class Value:
    __slots__ = (
        "data",
        "grad",
        "_children",
        "_local_grads",
    )  # Python optimization to reduce memory usage by defining fixed attributes

    def __init__(self, data, children=(), local_grads=()):
        self.data = data  # scalar value of this node calculated during forward pass
        self.grad = (
            0  # derivative of the loss w.r.t. this node, calculated in backward pass
        )
        self._children = children  # children of this node in the computation graph
        self._local_grads = (
            local_grads  # local derivative of this node w.r.t. its children
        )

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            self.data + other.data, (self, other), (1, 1)
        )  # forward pass: compute output value and store local gradients
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            self.data * other.data, (self, other), (other.data, self.data)
        )  # forward pass: compute output value and store local gradients
        return out

    def __pow__(self, other):
        out = Value(
            self.data**other, (self,), (other * self.data ** (other - 1),)
        )  # forward pass: compute output value and store local gradients
        return out

    def log(self):
        out = Value(
            math.log(self.data), (self,), (1 / self.data,)
        )  # forward pass: compute output value and store local gradients
        return out

    def exp(self):
        out = Value(
            math.exp(self.data), (self,), (math.exp(self.data),)
        )  # forward pass: compute output value and store local gradients
        return out

    def relu(self):
        out = Value(
            max(0, self.data), (self,), (float(self.data > 0),)
        )  # forward pass: compute output value and store local gradients
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1  # seed the gradient of the output node
        for v in reversed(topo):  # traverse the graph in reverse topological order
            for local_grad, child in zip(v._local_grads, v._children):
                child.grad += local_grad * v.grad  # chain rule to compute gradients


# ===Step 4: Initialize the parameters of the model, to store the knowledge of the model.
n_embd = 16  # embedding dimension
n_head = 4  # number of attention heads
n_layer = 1  # number of layers
block_size = 16  # maximum sequence length
head_dim = n_embd // n_head  # dimension of each head
matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)
]
state_dict = {
    "wte": matrix(vocab_size, n_embd),
    "wpe": matrix(block_size, n_embd),
    "lm_head": matrix(vocab_size, n_embd),
}
for i in range(n_layer):
    state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)
params = [
    p for mat in state_dict.values() for row in mat for p in row
]  # flatten params into a single list[Value]
print(f"num params: {len(params)}")


# ===STEP 5: Model Architecture Definition
#          Define the architecture of the model, including the embedding layer, multi-head self-attention mechanism, and feedforward neural network
#           - a stateless function mapping token sequence and parameters to logits over what comes next.
#               Follows GPT-2,with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
#               The model is a function that takes in a sequence of token indices and the model parameters, and outputs the logits for the next token in the sequence.
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_logit = max(logits)
    exp_logits = [math.exp(logit - max_logit) for logit in logits]
    sum_exp = sum(exp_logits)
    return [exp_logit / sum_exp for exp_logit in exp_logits]
