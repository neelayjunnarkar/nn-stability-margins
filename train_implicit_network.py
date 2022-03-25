import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from models.implicit_model import ImplicitModel
from activations import Tanh, LeakyReLU
from envs import InvertedPendulumEnv

device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

def train_loop(dataloader, model, loss_fn, optimizer):
    # 1. predict
    # 2. compute loss
    # 3. compute gradients through backpropogation
    # 4. update parameters
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        predictions = model(X)
        loss = loss_fn(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss = loss.item() / 1
            current = batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    model.eval()
    for (X, y) in dataloader:
        X = X.to(device)
        y = y.to(device)
        predictions = model(X)
        loss = loss_fn(predictions, y)
        test_loss += loss.item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss

# Setup true model

env = InvertedPendulumEnv
env_config = {
    "observation": "partial",
    "normed": True,
    "factor": 1,
    "dt": 0.02,
}
true_model = env(env_config)

# Setup model and optimizer

lr = 1e-4
batch_size = 128
epochs = 50

state_size = 2
nonlin_size = 4

N_train = 100000
N_test = 1000

model = ImplicitModel(state_size, nonlin_size, Tanh).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
loss_fn = torch.nn.MSELoss()

# Setup datasets

def create_dataset(size):
    init_states = torch.rand(size, true_model.state_size)
    init_states = 2*init_states - 1
    init_states = init_states * torch.from_numpy(true_model.state_space.high)

    # Set up next states 
    zero_input = torch.zeros(*true_model.action_space.shape)
    next_states = torch.zeros_like(init_states)
    for i in range(init_states.shape[0]):
        true_model.reset(init_states[i].numpy())
        true_model.step(zero_input.numpy())
        next_states[i] = torch.from_numpy(true_model.state)

    return init_states, next_states

train_dataset = TensorDataset(*create_dataset(N_train))
test_dataset  = TensorDataset(*create_dataset(N_test))

train_dataloader = DataLoader(train_dataset, batch_size)
test_dataloader  = DataLoader(test_dataset,  batch_size)

# Train Model

test_losses = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss = test_loop(test_dataloader, model, loss_fn)
    test_losses.append(test_loss)
print(test_losses)

plt.figure()
plt.semilogy(test_losses)
plt.show()

# Final test and display rollouts
N = 30
rollout_len = 100
control = torch.zeros(*true_model.action_space.shape)
init_states = torch.zeros(N, true_model.state_size)
for i in range(true_model.state_size):
    init_states[:, i] = torch.linspace(-true_model.state_space.high[i], true_model.state_space.high[i], N)

# Compute all rollouts
true_rollouts = torch.zeros(N, rollout_len, true_model.state_size)
model_rollouts = torch.zeros(N, rollout_len, true_model.state_size)
model.eval()
with torch.no_grad():
    for i in range(N):
        true_model.reset(init_states[i].numpy())
        true_rollouts[i, 0] = init_states[i]
        model_rollouts[i, 0] = init_states[i]
        for k in range(1, rollout_len):
            true_model.step(control.numpy(), fail_on_time_limit = False, fail_on_state_space = False)
            true_rollouts[i, k] = torch.from_numpy(true_model.state)
            model_rollouts[i, k] = model(model_rollouts[i, k-1].reshape(1, true_model.state_size))

    plt.figure()
    plt.subplot(211)
    for i in range(N):
        plt.plot(true_rollouts[i, :, 0].numpy(), 'tab:blue')
        plt.plot(model_rollouts[i, :, 0].numpy(), 'tab:orange')
    plt.title("theta")

    plt.subplot(212)
    for i in range(N):
        plt.plot(true_rollouts[i, :, 1].numpy(), 'tab:blue')
        plt.plot(model_rollouts[i, :, 1].numpy(), 'tab:orange')
    plt.title("theta dot")

    plt.show()

torch.save(model, 'inv_pend_model.pth')