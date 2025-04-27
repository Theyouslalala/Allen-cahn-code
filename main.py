import torch
from torch.optim.lr_scheduler import StepLR
from torch.autograd import grad
import math, time, os
import torch.nn as nn
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import json
import os
import torch
import pandas as pd


random.seed(1)
np.random.seed(1)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)


parser = argparse.ArgumentParser(description='Hyper parameters for modified PDE solver')
parser.add_argument('--e', type=int, default=2000, help='Epochs')
parser.add_argument('--d', type=int, default=3, help='Depth of the network')
parser.add_argument('--n', type=int, default=70, help='Width of the network')
parser.add_argument('--beta', type=int, default=0.01, help='Beta parameter')
parser.add_argument('--nx', type=int, default=256, help='Number of sampling points in each dimension')
parser.add_argument('--xi', type=float, default=1e-6, help='Threshold for early stopping')
args = parser.parse_args()


def save_results(params, pre_train_loss, train_loss, loss_r, loss_b, loss_v, model, U_pred):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)


    with open(os.path.join(results_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)


    np.savetxt(os.path.join(results_dir, "pre_train_loss.txt"), pre_train_loss)


    np.savetxt(os.path.join(results_dir, "train_loss.txt"), train_loss)
    np.savetxt(os.path.join(results_dir, "loss_r.txt"), loss_r)
    np.savetxt(os.path.join(results_dir, "loss_b.txt"), loss_b)
    np.savetxt(os.path.join(results_dir, "loss_v.txt"), loss_v)


    torch.save(model.state_dict(), os.path.join(results_dir, "model.pth"))


    np.savetxt(os.path.join(results_dir, "U_pred.txt"), U_pred.cpu().detach().numpy())


    loss_df = pd.DataFrame({
        'Total Loss': train_loss,
        'Residual Loss': loss_r,
        'Boundary Loss': loss_b,
        'Volume Loss': loss_v
    })
    loss_df.to_excel(os.path.join(results_dir, "all_losses.xlsx"), index=False)

    print(f"Results saved in {results_dir} directory.")


def GetGradients(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f),
                create_graph=True, only_inputs=True, allow_unused=True)[0]


def PHI_0(x):
    u0 = np.sin(np.pi * x)
    #u0 = -np.sin(np.pi * x)

    return u0


def errorFun(output, target, params):
    error = output - target
    error = torch.sqrt(torch.mean(error * error))
    ref = torch.sqrt(torch.mean(target * target))
    return error / (ref + params["minimal"])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class Net(torch.nn.Module):
    def __init__(self, params, device):
        super(Net, self).__init__()
        self.params = params
        self.device = device
        self.linearIn = nn.Linear(self.params["d"], self.params["width"])
        nn.init.xavier_normal_(self.linearIn.weight)
        nn.init.constant_(self.linearIn.bias, 0)
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            layer = nn.Linear(self.params["width"], self.params["width"])
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
            self.linear.append(layer)
        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])
        nn.init.xavier_normal_(self.linearOut.weight)
        nn.init.constant_(self.linearOut.bias, 0)

    def shift_scale_layer(self, x):
        x_min = torch.min(x)
        x_max = torch.max(x)
        if x_min != x_max:
            x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1
        else:
            x_scaled = torch.zeros_like(x)
        return x_scaled

    def forward(self, X):
        x = torch.relu(self.linearIn(X))
        for layer in self.linear:
            x = torch.relu(layer(x))
        x = self.linearOut(x)

        x = self.shift_scale_layer(x)

        return x


def boundary_loss(model, device, params):
    x_b = torch.tensor([-1, 1], device=device).float().unsqueeze(1)
    x_b.requires_grad = True
    U_b_pred = model(x_b)
    U_b_x = GetGradients(U_b_pred, x_b)
    boundary_loss_value = torch.mean(U_b_x ** 2)
    return boundary_loss_value


def pre_train(model, device, params, optimizer, scheduler):


    Loss = []

    for step in range(params["pre_trainstep"]):

        x = np.linspace(-1, 1, params["nx"])
        X_tensor = torch.from_numpy(x[:, None]).float().to(device)
        u0 = PHI_0(x).flatten()[:, None]
        U0 = torch.from_numpy(u0).float().to(device)

        U0_pred = model(X_tensor)

        model.zero_grad()

        loss = torch.mean(torch.square(U0_pred - U0))
        Loss.append(loss.cpu().detach().numpy())

        if step % params["pre_step"] == 0:
            print('Epoch: %d, Loss: %.3e' % (step, loss))

        loss.backward()
        optimizer.step()
        scheduler.step()

    return model, Loss


def train(model, device, params, optimizer, scheduler):
    x_train = np.linspace(-1, 1, params["nx"])
    X_train_tensor = torch.from_numpy(x_train[:, None]).float().to(device)
    X_train_tensor.requires_grad = True

    # 计算初值u0
    u0 = PHI_0(x_train).flatten()[:, None]
    U0 = torch.from_numpy(u0).float().to(device)

    start_time = time.time()
    total_start_time = start_time
    Loss = []
    Loss_r = []
    Loss_b = []
    Loss_v = []

    min_target_var = torch.tensor(1.0).to(device)

    lambda_var = 1.5  # 方差约束权重

    for step in range(params["trainstep"]):
        U_pred = model(X_train_tensor)
        model.zero_grad()
        U_x = GetGradients(U_pred, X_train_tensor)[:, 0:1]

        Res = 0.5 * params["beta"] ** 2 * U_x ** 2 + 0.25 * (U_pred ** 2 - 1) ** 2
        dx = (1 - (-1)) / params["nx"]  # 即 dx = 2 / params["nx"]
        loss_res = torch.sum(Res) * dx

        loss_boundary = boundary_loss(model, device, params)

        predicted_var = torch.var(U_pred)
        variance_diff = min_target_var - predicted_var

        variance_constraint = torch.relu(variance_diff) ** 2



        loss = loss_res + loss_boundary + lambda_var * variance_constraint

        Loss.append(loss.cpu().item())
        Loss_r.append(loss_res.cpu().item())
        Loss_b.append(loss_boundary.cpu().item())
        Loss_v.append(variance_constraint.cpu().item())

        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f'Epoch: {step}, Time: {elapsed:.2f}, Loss: {loss.item():.3e}, Loss_r: {loss_res.item():.3e}, Loss_b:{loss_boundary.item():.3e},Loss_v:{variance_constraint.item():.3e}', )
            # print(f'Epoch: {step}, Time: {elapsed:.2f}, Loss: {loss.item():.3e}, Loss_r: {loss_res.item():.3e}, Loss_b {loss_boundary.item():.3e}',)
            start_time = time.time()

        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"Training finished. Total time: {time.time() - total_start_time:.2f} seconds")
    return Loss, Loss_r, Loss_b, Loss_v, U_pred

def plot_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()


def plot_results(model, device, params):
    x_test = np.linspace(-1, 1, params["nx"])
    X_test_tensor = torch.from_numpy(x_test[:, None]).float().to(device)
    U_pred_test = model(X_test_tensor).cpu().detach().numpy().flatten()

    plt.figure(figsize=(10, 5))
    plt.plot(x_test, U_pred_test)
    plt.title('Predicted Wave Function after Training')
    plt.xlabel('x')
    plt.ylabel('Wave Function')
    plt.show()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = dict()
    params["d"] = 1
    params["interval"] = 1
    params["nx"] = args.nx
    params["ny"] = args.nx
    params["width"] = args.n
    params["depth"] = args.d
    params["dd"] = 1
    params["lr"] = 0.001
    params["beta"] = args.beta
    params["xi"] = args.xi
    params["trainstep"] = args.e
    params["pre_trainstep"] = 1000
    params["pre_step"] = 100
    params["minimal"] = 1e-14
    params["step_size"] = 100
    params["gamma"] = 0.99

    model = Net(params, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = StepLR(optimizer, step_size=params["step_size"], gamma=params["gamma"])

    model_pre, pre_train_loss = pre_train(model, device, params, optimizer, scheduler)

    x_pre = np.linspace(-1, 1, params["nx"])
    X_pre_tensor = torch.from_numpy(x_pre[:, None]).float().to(device)
    U_pre = model_pre(X_pre_tensor)

    pretrain_dir = "pretrain_results"
    os.makedirs(pretrain_dir, exist_ok=True)

    np.savetxt(os.path.join(pretrain_dir, "U_pretrain.txt"), U_pre.cpu().detach().numpy())

    np.savetxt(os.path.join(pretrain_dir, "pretrain_loss.txt"), pre_train_loss)

    torch.save(model_pre.state_dict(), os.path.join(pretrain_dir, "pretrain_model.pth"))

    plot_results(model, device, params)
    train_loss, loss_r, loss_b, loss_v, U_pred = train(model_pre, device, params, optimizer, scheduler)
    plot_loss(pre_train_loss)
    plot_loss(train_loss)
    plot_results(model, device, params)
    print(f"The number of parameters is {count_parameters(model)}")

    save_results(params, pre_train_loss, train_loss, loss_r, loss_b, loss_v, model, U_pred)


if __name__ == "__main__":
    main()
