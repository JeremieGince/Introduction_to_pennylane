import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import pennylane as qml
# link: https://pennylane.ai/qml/demos/tutorial_qnn_module_torch.html


# Fixing the dataset and problem
X, y = make_moons(n_samples=200, noise=0.1)
y_ = torch.unsqueeze(torch.tensor(y), 1)  # used for one-hot encoded labels
y_hot = torch.scatter(torch.zeros((200, 2)), 1, y_, 1)

c = ["#1f77b4" if y_ == 0 else "#ff7f0e" for y_ in y]  # colours for each class
plt.axis("off")
plt.scatter(X[:, 0], X[:, 1], c=c)
plt.show()

# Defining a QNode
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def qnode(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


# Interfacing with Torch
n_layers = 6
weight_shapes = {"weights": (n_layers, n_qubits)}
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)


# Creating a hybrid model
clayer_1 = torch.nn.Linear(2, 2)
clayer_2 = torch.nn.Linear(2, 2)
softmax = torch.nn.Softmax(dim=1)
layers = [clayer_1, qlayer, clayer_2, softmax]
base_model = torch.nn.Sequential(*layers)


# Training the model
def train_model(_model):
    global X
    global y_hot

    # _model.cuda()

    opt = torch.optim.SGD(_model.parameters(), lr=0.2)
    loss = torch.nn.MSELoss()

    X = torch.tensor(X, requires_grad=True).float()
    y_hot = torch.tensor(y_hot)

    batch_size = 5
    batches = 200 // batch_size

    data_loader = torch.utils.data.DataLoader(
        list(zip(X, y_hot)), batch_size=5, shuffle=True, drop_last=True
    )

    epochs = 6

    for epoch in range(epochs):

        running_loss = 0

        for xs, ys in data_loader:
            # xs, ys = xs.cuda(), ys.cuda()
            opt.zero_grad()
            # print(xs.device, ys.device)
            loss_evaluated = loss(_model(xs), ys)
            loss_evaluated.backward()

            opt.step()

            running_loss += loss_evaluated

        avg_loss = running_loss / batches
        print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))

    y_pred = _model(X)
    predictions = torch.argmax(y_pred, axis=1).detach().numpy()

    correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, y)]
    accuracy = sum(correct) / len(correct)
    print(f"Accuracy: {accuracy * 100}%")


# Creating non-sequential models
class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clayer_1 = torch.nn.Linear(2, 4)
        self.qlayer_1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer_2 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.clayer_2 = torch.nn.Linear(4, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.clayer_1(x)
        x_1, x_2 = torch.split(x, 2, dim=1)
        x_1 = self.qlayer_1(x_1)
        x_2 = self.qlayer_2(x_2)
        x = torch.cat([x_1, x_2], axis=1)
        x = self.clayer_2(x)
        return self.softmax(x)


# Creating non-sequential models
class ClassicalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clayer_1 = torch.nn.Linear(2, 4)
        self.clayer_2 = torch.nn.Linear(2, 2)
        self.clayer_3 = torch.nn.Linear(2, 2)
        self.clayer_4 = torch.nn.Linear(4, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.clayer_1(x)
        x_1, x_2 = torch.split(x, 2, dim=1)
        x_1 = self.clayer_2(x_1)
        x_2 = self.clayer_3(x_2)
        x = torch.cat([x_1, x_2], axis=1)
        x = self.clayer_4(x)
        return self.softmax(x)


if __name__ == '__main__':
    q_model = HybridModel()
    c_model = ClassicalModel()

    print("base-q-model")
    train_model(base_model)
    print('-' * 75)
    print("q-model")
    train_model(q_model)
    print('-'*75)
    print("c-model")
    train_model(c_model)
    print('-'*75)
