import dataclasses
import os
import traceback
from datetime import datetime
from typing import Dict, Optional, Iterator, Tuple

import torch
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from torch.nn import MSELoss, Parameter
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.linear_model import DeepLipschitzLinearResNet, DeepLipschitzSequential, DeepLipschitzSequentialStack

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


@dataclasses.dataclass()
class FlossingConfig:
    enabled: bool = False
    enable_logging: bool = True
    flossing_frequency: Optional[int] = None
    flossing_le: int = 1
    num_sample_trajectories: int = 300
    weight: float = 0.05
    offset: float = 0.0
    stop_criteria: float = -0.2
    conditioning_steps = 15


def initial_train_condition(training_loader, optimizer, model, tb_writer,
                            flossing_config: FlossingConfig,
                            logging_frequency=100):
    print(
        "Initially generating a conditioning of the model to move the Lyapunov exponents to a regime before the actual training")

    running_loss = 0.

    running_max_le = None

    lyponov_exponents = []

    stop = False

    global_step = 0

    for epoch_index in range(flossing_config.conditioning_steps):
        print(f'epoch {epoch_index}:')

        average_running_max = 0

        batch_numbers = 0

        for batch_index, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            if hasattr(model, 'in_features'):
                inputs = inputs.reshape((-1, model.in_features))

            # Zero your gradients for every batch!
            optimizer.zero_grad(set_to_none=True)

            le, outputs = model.calculate_lyapunov_spectrum(inputs, nle=flossing_config.flossing_le,
                                                            n_random_samples=flossing_config.num_sample_trajectories,
                                                            normalization_frequency=flossing_config.flossing_frequency)

            lyponov_exponents.append(le.detach())

            current_max = le.max().item()

            if running_max_le is None:
                running_max_le = current_max
            else:
                running_max_le = max(running_max_le, current_max)

            average_running_max += current_max
            batch_numbers += 1

            flossing_loss = torch.square(le + flossing_config.offset).mean()

            loss = flossing_loss

            loss.backward()

            # Adjust learning weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            # print("Current loss:", loss.item())
            if global_step % logging_frequency == logging_frequency - 1 or stop:
                last_loss = running_loss / logging_frequency  # loss per batch

                tb_x = global_step
                tb_writer.add_scalar('Conditioned Loss/Train', last_loss, tb_x)

                tb_writer.add_scalar('Conditioned Loss/Flossing Max', running_max_le, tb_x)

                current_average = average_running_max / batch_numbers
                tb_writer.add_scalar('Conditioned Loss/Flossing Average', running_max_le, current_average)

                print(
                    '  batch {} step {} flossing (max: {} average: {}): {}'.format(batch_index + 1, global_step + 1,
                                                                                   running_max_le, current_average,
                                                                                   last_loss))

                running_max_le = None
                running_loss = 0.

                log_all(tb_writer, model.named_parameters(), tb_x, prefix="Conditioned")

            global_step += 1

        epoch_average = average_running_max / batch_numbers

        print(f'epoch {epoch_index}: average max {epoch_average}')

        if abs(epoch_average) <= abs(flossing_config.stop_criteria):
            print(
                f"Early termination since the max Lyapunov exponent within the stopping criteria {flossing_config.stop_criteria} max: {epoch_average}")
            break

    print("Finished flossing conditioning")

    return lyponov_exponents


def train_one_epoch(training_loader, optimizer, model, loss_fn, epoch_index, tb_writer, flossing_config: FlossingConfig,
                    logging_frequency=100):
    running_loss = 0.
    last_loss = 0.

    running_output_loss = 0.
    running_flossing_loss = 0.

    running_max_le = None

    lyponov_exponents = []

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

        if flossing_config.enabled or flossing_config.enable_logging:
            # FIXME gradient issue with this and the forward function, disabling training
            le, outputs = model.calculate_lyapunov_spectrum(inputs, nle=flossing_config.flossing_le,
                                                            n_random_samples=flossing_config.num_sample_trajectories,
                                                            normalization_frequency=flossing_config.flossing_frequency)

            lyponov_exponents.append(le.detach())

            if running_max_le is None:
                running_max_le = le.max().item()
            else:
                running_max_le = max(running_max_le, le.max().item())

            flossing_loss = torch.square(le + flossing_config.offset).mean()

            output_loss = loss_fn(outputs, labels)

            if flossing_config.weight > 0 and flossing_config.enabled and flossing_loss < flossing_config.stop_criteria:
                loss = flossing_config.weight * flossing_loss + output_loss
            else:
                loss = output_loss
        else:
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

        if flossing_config.enabled or flossing_config.enable_logging:
            running_output_loss += output_loss.item()
            running_flossing_loss += flossing_loss.item()
        else:
            running_output_loss += loss.item()

        # print("Current loss:", loss.item())
        if i % logging_frequency == logging_frequency - 1:
            last_loss = running_loss / logging_frequency  # loss per batch

            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/Train', last_loss, tb_x)

            last_output_loss = running_output_loss / logging_frequency

            if flossing_config.enabled or flossing_config.enable_logging:
                last_flossing_loss = running_flossing_loss / logging_frequency

                tb_writer.add_scalar('Loss/Flossing', last_flossing_loss, tb_x)
                tb_writer.add_scalar('Loss/Flossing Max', running_max_le, tb_x)

                print('  batch {} loss: {} flossing (max: {}): {} output: {}'.format(i + 1, last_loss, running_max_le,
                                                                                     last_flossing_loss,
                                                                                     last_output_loss))

            else:
                print('  batch {} loss: {}'.format(i + 1, last_loss))

            tb_writer.add_scalar('Loss/Output', last_output_loss, tb_x)

            running_max_le = None
            running_loss = 0.
            running_flossing_loss = 0.0
            running_output_loss = 0.0

            log_all(tb_writer, model.named_parameters(), tb_x)

            if hasattr(model, 'A'):
                alpha_vals = alpha_values(model)

                tb_writer.add_scalars('Training/Alpha', alpha_vals, tb_x, tb_x)

    return last_loss, lyponov_exponents


@torch.no_grad()
def log_all(summary_writer, parameters: Iterator[Tuple[str, Parameter]], step, prefix=''):
    for n, param in parameters:
        if param.numel() == 1:
            fnc = summary_writer.add_scalar
        else:
            fnc = summary_writer.add_histogram

        if param.requires_grad and param.grad is not None:
            fnc(f"{prefix} gradients/{n}", param.grad, global_step=step)

        fnc(f"{prefix} values/{n}", param, global_step=step)


def train(x, y, model: DeepLipschitzLinearResNet, flossing_config: Optional[FlossingConfig] = None,
          learning_prefix='sine', batch_size=64, lr=1e-3,
          termination_error=5e-3, epochs=100, theoretical_lower=0, logging_frequency=100,
          logging_folder=None):
    if flossing_config is None:
        flossing_config = FlossingConfig(enabled=False)

    if logging_folder is not None:
        save_folder = logging_folder
    else:
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_folder = f'../runs/{learning_prefix}_trainer_{timestamp}'

    writer = SummaryWriter(save_folder)

    hparam_dict = {
        'flossing': flossing_config.enabled,
        'weight': flossing_config.weight,
        'num_sample_trajectories': flossing_config.num_sample_trajectories,
        'flossing_le': flossing_config.flossing_le,
        'offset': flossing_config.offset,
    }

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

    error = None

    running_lyapunov_exponents = []

    conditioned_lyapunov_exponents = []

    if flossing_config.conditioning_steps > 0 and flossing_config.enabled:
        try:
            conditioned_lyapunov_exponents = initial_train_condition(training_loader, optimizer, model, writer,
                                                                     flossing_config, logging_frequency)

            if len(conditioned_lyapunov_exponents) > 0:
                conditioned_lyapunov_exponents = torch.vstack(conditioned_lyapunov_exponents).tolist()
        except Exception as e:
            error = e

            return error, save_folder, best_loss, best_epoch, losses, running_lyapunov_exponents, conditioned_lyapunov_exponents

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)

        try:
            avg_loss, lyponov_exponents = train_one_epoch(training_loader, optimizer, model, loss_fn, epoch_number,
                                                          writer, flossing_config,
                                                          logging_frequency=logging_frequency)
        except Exception as e:
            traceback.print_exc()
            error = e
            break

        running_lyapunov_exponents.extend(lyponov_exponents)

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

            writer.add_hparams(hparam_dict, metric_dict={'hparam/best_loss': best_loss}, global_step=epoch_number)

        epoch_number += 1

    if len(running_lyapunov_exponents) > 0:
        running_lyapunov_exponents = torch.vstack(running_lyapunov_exponents).tolist()

    return error, save_folder, best_loss, best_epoch, losses, running_lyapunov_exponents, conditioned_lyapunov_exponents


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

    _, _, _, _, losses, _, _ = train(x, y, model, learning_prefix='sine', batch_size=batch, termination_error=1e-4,
                                     lr=1e-4,
                                     theoretical_lower=theoretical_lower,
                                     epochs=epochs)

    return losses


def sine_training_flossing(L=10, hidden=64, epochs=20, device_id: int = 0):
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

    flossing_config = FlossingConfig(enabled=True, flossing_frequency=1, weight=0.1, enable_logging=True)

    _, _, _, _, losses, _, _ = train(x, y, model, flossing_config=flossing_config, learning_prefix='sine_flossing',
                                     batch_size=batch, termination_error=1e-4, lr=1e-4,
                                     theoretical_lower=theoretical_lower,
                                     epochs=epochs)

    return losses


def sine_training_grouped(L=2, hidden=64, L_int=4, epochs=20, device_id: int = 0):
    torch.manual_seed(0)
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

    dtype = torch.float32

    print("Running on device", device)

    input_features = 1
    output_features = input_features

    model = DeepLipschitzSequentialStack(input_features, output_features, num_layers=L, num_hidden=hidden,
                                         num_interior_layers=L_int, device=device)

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

    _, _, _, _, losses, _, _ = train(x, y, model, learning_prefix='sine_stacked', batch_size=batch,
                                     termination_error=1e-4,
                                     lr=1e-4,
                                     theoretical_lower=theoretical_lower,
                                     epochs=epochs)

    return losses


if __name__ == '__main__':
    # sine_training(L=10)
    # sine_training_grouped(L=5)
    sine_training_flossing(L=30)
    # main()
