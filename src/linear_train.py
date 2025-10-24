import os
from datetime import datetime
from typing import Dict

import torch
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from torch.nn import MSELoss
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.linear_model import DeepLipschitzLinearResNet, DeepLipschitzSequential

DPI = 600

torch.backends.fp32_precision = "tf32"
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cudnn.rnn.fp32_precision = "tf32"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


# torch.autograd.set_detect_anomaly(True)

@torch.no_grad()
def alpha_values(model: DeepLipschitzLinearResNet) -> Dict[str, float]:
    output = dict()

    output['A'] = model.A.alpha.item()
    output['B'] = model.B.alpha.item()

    print("A", output['A'])
    print("B", output['B'])
    for i, l in enumerate(model.layers):
        val = l.alpha.item()

        output[f'C{i + 1}'] = val
        print(i + 1, val)

    return output


def train_one_epoch(training_loader, optimizer, model, loss_fn, epoch_index, tb_writer, logging_frequency=100):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        if hasattr(model, 'in_features'):
            inputs = inputs.reshape((-1, model.in_features))
            labels = labels.reshape((-1, model.out_features))

        # Zero your gradients for every batch!
        optimizer.zero_grad(set_to_none=True)

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        # print("Current loss:", loss.item())
        if i % logging_frequency == logging_frequency - 1:
            last_loss = running_loss / logging_frequency  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/Train', last_loss, tb_x)
            running_loss = 0.

            if hasattr(model, 'A'):
                alpha_vals = alpha_values(model)

                tb_writer.add_scalars('Training/Alpha', alpha_vals, tb_x, tb_x)

    return last_loss


def train(x, y, model: DeepLipschitzLinearResNet, learning_prefix='sine', batch_size=64, lr=1e-3,
          termination_error=5e-3, epochs=100, theoretical_lower=0, logging_frequency=100):
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_folder = f'../runs/{learning_prefix}_trainer_{timestamp}'

    writer = SummaryWriter(save_folder)

    # writer.add_graph(model, torch.randn((10, model.in_features), device=model.device))

    epoch_number = 0

    loss_fn = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=True)

    dataset = TensorDataset(x, y)

    training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = None
    best_epoch = None

    losses = []

    # theoretical_lower = 135.021966261
    # theoretical_lower = 0

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(training_loader, optimizer, model, loss_fn, epoch_number, writer,
                                   logging_frequency=logging_frequency)

        print('LOSS train {}'.format(avg_loss))

        losses.append(avg_loss)

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training',
                           {'Training': avg_loss},
                           epoch_number + 1)
        writer.flush()

        with torch.no_grad():
            if hasattr(model, 'in_features'):
                y_pred = model(x.reshape(-1, model.in_features))

                plt.plot(x.cpu(), y.cpu(), label='Ground Truth')
                plt.plot(x.cpu(), y_pred.cpu(), label='Prediction')
                plt.legend()
                plt.tight_layout()

                plt.savefig(os.path.join(save_folder, f'truth{epoch}.png'), dpi=DPI)
                plt.close()
            else:
                y_pred = model(x)

            dy = torch.diff(y_pred, dim=0)
            dx = torch.diff(x, dim=0)

            L = torch.linalg.norm(dy) / torch.linalg.norm(dx)
            L = L.max().item()

            # relative_error = (L - model.lipschitz_constant) / model.lipschitz_constant

            writer.add_scalar("Training/Lipschitz Constant", L, epoch_number + 1)

            # assert relative_error <= 1.25, f"The Lipschitz constraint was invalidated with {L} {model.lipschitz_constant} {relative_error}"

            print("Lipschitz constant", L)

        assert avg_loss >= theoretical_lower, "Problem with the lipschitz network since cannot be lower than theoretical lowest"

        if avg_loss < termination_error:
            break

        if best_loss is None or best_loss > avg_loss:
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_folder, 'best.pt'))
            best_loss = avg_loss
            print(f"New best loss: {best_loss} at {best_epoch}")

            writer.add_scalar('Training/Best Epoch', best_epoch + 1, epoch_number + 1)
            writer.add_scalar('Training/Best Loss', best_loss, epoch_number + 1)

        epoch_number += 1

    return save_folder, best_loss, best_epoch, losses


def print_num_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def sine_training(L=10, hidden=64, epochs=20, device_id: int = 0):
    torch.manual_seed(0)
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

    dtype = torch.float32

    print("Running on device", device)

    input_features = 1
    output_features = input_features

    model = DeepLipschitzSequential(input_features, output_features, (hidden,) * L, device=device)

    check_gradients = False

    if check_gradients:
        for name, p in model.named_parameters():
            def function(grad):
                if grad.isnan().any():
                    print("GRADIENT IS NAN IN {}".format(name))

            p.register_post_accumulate_grad_hook(function)

    model = torch.compile(model, mode='max-autotune')

    print(model)
    print(print_num_parameters(model))

    batch = 64

    x = torch.linspace(-10, 10, 100000 * input_features, device=device, dtype=dtype).reshape((-1, input_features))

    variation = torch.arange(input_features, device=device, dtype=dtype)

    y = torch.sin(x + variation)
    theoretical_lower = 0

    _, _, _, losses = train(x, y, model, learning_prefix='sine', batch_size=batch, termination_error=1e-4, lr=1e-4,
                            theoretical_lower=theoretical_lower,
                            epochs=epochs)

    return losses


def grid_search(L=15, hidden=(8, 16, 32, 64, 128)):
    pass


if __name__ == '__main__':
    grid_search()
    # sine_training(L=10)
    # main()
