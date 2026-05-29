import statistics
# import time больше не нужен

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train():
    dataloader = DataLoader(
        prepare_data(),
        batch_size=256,
        shuffle=True,
        pin_memory=True, # faster host -> device transfer
    )

    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).cuda().train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses_history = []
    forward_times = []
    backward_times = []

    for batch_idx, (data, target) in enumerate(dataloader):

        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # создаем noise сразу на GPU
        noise = torch.randn_like(data)

        data = data + noise

        optimizer.zero_grad(set_to_none=True)

        # корректный timing GPU операций
        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)

        start_fwd.record()

        output = model(data)
        loss = criterion(output, target)

        end_fwd.record()

        start_bwd = torch.cuda.Event(enable_timing=True)
        end_bwd = torch.cuda.Event(enable_timing=True)

        start_bwd.record()

        loss.backward()
        optimizer.step()

        end_bwd.record()

        torch.cuda.synchronize()

        forward_times.append(start_fwd.elapsed_time(end_fwd))
        backward_times.append(start_bwd.elapsed_time(end_bwd))

        # сохраняем scalar, а не computation graph
        losses_history.append(loss.item())

        print(f"Batch {batch_idx} loss: {loss.item():.4f}")

        # empty_cache внутри train loop удаляем
        # torch.cuda.empty_cache()

    print(
        f"Epoch finished, avg forward time is {statistics.mean(forward_times)}, "
        f"avg backward time is {statistics.mean(backward_times)}"
    )


if __name__ == "__main__":
    train()
