"""Training utilities for the pure-Jittor GMT implementation.

The filename is kept for import compatibility with earlier scripts.
"""

import math
import os

import jittor as jt
from tensorboardX import SummaryWriter
from tqdm import tqdm

from train._utils import assemble_F_fast, hexahedron, isotropic_elastic_tensor
from train.dataset import SparseDataset, iter_sparse_batches
from train.model import GMT


def residual_loss(residual: jt.Var, coord: jt.Var, batch_size: int, logarithmic=True):
    # Match the original PyTorch trainer: accumulate residual norms in float64.
    # Squaring a large float32 residual can overflow before gradient clipping runs.
    total = jt.array(0.0).float64()
    batch_ids = coord[:, 0]
    for batch_id in range(batch_size):
        selected = residual[batch_ids == batch_id].float64()
        norm = jt.sqrt((selected * selected).sum(dim=0).sum(dim=0) + 1e-30)
        if logarithmic:
            norm = jt.log(norm) / math.log(10.0)
        total = total + norm.mean()
    return total / batch_size


def _assert_finite_tensor(name: str, value: jt.Var):
    if not bool(value.isfinite().all().item()):
        raise FloatingPointError(
            f"Non-finite values first detected in {name}. "
            "The optimizer update was not applied."
        )


def _named_optimizer_gradients(model, optimizer):
    names = {id(param): name for name, param in model.named_parameters()}
    for group_id, group in enumerate(optimizer.param_groups):
        for param_id, (param, grad) in enumerate(
            zip(group["params"], group.get("grads", []))
        ):
            if param.is_stop_grad():
                continue
            name = names.get(id(param), f"param_group_{group_id}.{param_id}")
            yield name, grad


def _assert_finite_gradients(model, optimizer):
    named_grads = list(_named_optimizer_gradients(model, optimizer))
    if not named_grads:
        return
    checks = jt.concat(
        [grad.isfinite().all().reshape(1) for _, grad in named_grads], dim=0
    )
    if bool(checks.all().item()):
        return
    for name, grad in named_grads:
        if not bool(grad.isfinite().all().item()):
            raise FloatingPointError(
                f"Non-finite gradient first detected in parameter '{name}'. "
                "The optimizer update was not applied."
            )


def _assert_finite_parameters(model):
    named_parameters = model.named_parameters()
    checks = jt.concat(
        [param.isfinite().all().reshape(1) for _, param in named_parameters], dim=0
    )
    if bool(checks.all().item()):
        return
    for name, param in named_parameters:
        if not bool(param.isfinite().all().item()):
            raise FloatingPointError(
                f"Non-finite parameter first detected in '{name}' after optimizer.step()."
            )


def _clip_grad_norm_float64(model, optimizer, max_norm: float) -> float:
    """Clip gradients without overflowing Jittor's float32 global norm."""
    named_grads = list(_named_optimizer_gradients(model, optimizer))
    total_squared = jt.array(0.0).float64()
    for _, grad in named_grads:
        grad64 = grad.float64()
        total_squared = total_squared + (grad64 * grad64).sum()
    total_norm = jt.sqrt(total_squared)
    coefficient = jt.minimum(
        jt.array(float(max_norm)).float64() / (total_norm + 1e-12),
        jt.array(1.0).float64(),
    ).float32()
    for _, grad in named_grads:
        grad.update(grad * coefficient)
    return float(total_norm.item())


def _set_lr(optimizer, lr: float):
    optimizer.lr = lr
    for group in optimizer.param_groups:
        group["lr"] = lr


def cosine_lr(base_lr: float, epoch: int, max_epoch: int) -> float:
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, max_epoch)))


def validate(model, dataset, cfg, Ke, Fe):
    model.eval()
    total = 0.0
    count = 0
    with jt.no_grad():
        for batch in iter_sparse_batches(
            dataset,
            int(cfg.batch_size),
            shuffle=False,
            rank=jt.rank,
            world_size=jt.world_size,
        ):
            F = assemble_F_fast(batch["node_index"], Fe, batch["feature"].shape[0])
            _, residual = model(batch, Ke, F)
            loss = residual_loss(residual, batch["coord"], int(cfg.batch_size), False)
            total += float(loss.item())
            count += 1
    stats = jt.array([total, float(count)]).float64()
    if jt.in_mpi:
        stats = stats.mpi_all_reduce()
    model.train()
    return float(stats[0].item()) / max(1.0, float(stats[1].item()))


def train(cfg):
    use_cuda = str(cfg.device).lower() in {"gpu", "cuda", "cuda:0"} and jt.has_cuda
    jt.flags.use_cuda = int(use_cuda)
    jt.set_global_seed(0)
    visible_devices = [
        device
        for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if device.strip()
    ]
    if use_cuda and not jt.in_mpi and len(visible_devices) > 1:
        print(
            "WARNING: multiple GPUs are visible, but only one Python process was "
            "started. Use mpirun -np <gpu_count> for Jittor data parallel training."
        )
    local_world_size = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", "1"))
    if (
        use_cuda
        and jt.in_mpi
        and visible_devices
        and local_world_size > len(visible_devices)
    ):
        raise RuntimeError(
            f"MPI started {local_world_size} local processes, but only "
            f"{len(visible_devices)} CUDA devices are visible."
        )

    train_ds = SparseDataset(cfg.train_data_path)
    val_paths = getattr(cfg, "val_data_path", getattr(cfg, "vail_data_path", []))
    val_ds = SparseDataset(val_paths)
    if len(train_ds) == 0:
        raise RuntimeError(
            "No training .npz files were found. Update train_data_path in the config."
        )

    model = GMT(cfg)
    if cfg.pre_train:
        model.load_state_dict(jt.load(str(cfg.pre_train)))

    optimizer = jt.optim.AdamW(
        model.parameters(), lr=float(cfg.learning_rate), weight_decay=0.01
    )
    C = isotropic_elastic_tensor(1.0, 0.3)
    Ke, Fe = hexahedron(int(cfg.resolution), C)

    log_dir = os.path.join(cfg.output_path, cfg.logger.path, cfg.logger.version)
    checkpoint_dir = os.path.join(cfg.output_path, cfg.checkpoint.path)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    is_main_process = jt.rank == 0
    writer = SummaryWriter(log_dir) if is_main_process else None
    if is_main_process:
        print(
            f"Jittor distributed={jt.in_mpi} world_size={jt.world_size} "
            f"per_gpu_batch_size={cfg.batch_size} "
            f"global_batch_size={int(cfg.batch_size) * jt.world_size}"
        )

    best_val = float("inf")
    global_step = 0
    base_lr = float(cfg.learning_rate)
    warmup_steps = max(0, int(getattr(cfg, "warmup_steps", 0)))
    gradient_clip_norm = float(getattr(cfg, "gradient_clip_norm", 1.0))
    for epoch in range(int(cfg.max_epoch)):
        model.train()
        epoch_lr = cosine_lr(base_lr, epoch, int(cfg.max_epoch))
        progress = tqdm(
            iter_sparse_batches(
                train_ds,
                int(cfg.batch_size),
                shuffle=True,
                rank=jt.rank,
                world_size=jt.world_size,
                seed=epoch,
            ),
            desc=f"Epoch {epoch + 1}/{cfg.max_epoch}",
            total=len(train_ds) // (int(cfg.batch_size) * jt.world_size),
            disable=not is_main_process,
        )
        for batch in progress:
            warmup_scale = (
                min(1.0, float(global_step + 1) / warmup_steps)
                if warmup_steps
                else 1.0
            )
            lr = epoch_lr * warmup_scale
            _set_lr(optimizer, lr)
            F = assemble_F_fast(batch["node_index"], Fe, batch["feature"].shape[0])
            _, residual = model(batch, Ke, F)
            _assert_finite_tensor("model residual before loss", residual)
            loss = residual_loss(residual, batch["coord"], int(cfg.batch_size), True)
            _assert_finite_tensor("training loss before backward", loss)
            optimizer.zero_grad()
            optimizer.backward(loss)
            _assert_finite_gradients(model, optimizer)
            grad_norm = _clip_grad_norm_float64(model, optimizer, gradient_clip_norm)
            optimizer.step()
            if global_step < 10:
                _assert_finite_parameters(model)
            value = float(loss.item())
            if writer:
                writer.add_scalar("train/residual_log10", value, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/gradient_norm", grad_norm, global_step)
            progress.set_postfix(loss=f"{value:.5f}", grad=f"{grad_norm:.3e}")
            global_step += 1

        val_loss = validate(model, val_ds, cfg, Ke, Fe) if len(val_ds) else float("nan")
        if writer:
            writer.add_scalar("validation/residual", val_loss, epoch)
        if is_main_process:
            jt.save(model.state_dict(), os.path.join(checkpoint_dir, "last.pkl"))
            if val_loss < best_val:
                best_val = val_loss
                jt.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pkl"))
            print(f"epoch={epoch + 1} val_residual={val_loss:.6e}")

    if writer:
        writer.close()
    return model
