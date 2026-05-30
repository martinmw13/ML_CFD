"""Fully connected autoencoders mirroring the spatial/temporal notebook setups."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlcfd.config.schemas import AutoencoderModelConfig
from mlcfd.logging_config import get_logger
from mlcfd.models.base import TrainableModel
from mlcfd.orientation import SnapshotOrientation, orient

LOGGER = get_logger("models")


class _LinearAutoencoderModule(nn.Module):
    """Encoder/decoder MLP with ReLU between layers and no tail activation."""

    def __init__(self, dims: list[int]) -> None:
        """Build mirrored encoder/decoder stacks.

        Args:
            dims: Sizes ``[input, h1, ..., latent]`` describing the encoder path.
        """
        super().__init__()
        encoder_layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                encoder_layers.append(nn.ReLU())
        decoder_layers: list[nn.Module] = []
        rev = list(reversed(dims))
        for idx in range(len(rev) - 1):
            decoder_layers.append(nn.Linear(rev[idx], rev[idx + 1]))
            if idx < len(rev) - 2:
                decoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return decoded outputs and latent codes."""
        latent = self.encoder(batch)
        recon = self.decoder(latent)
        return recon, latent


class LinearAutoencoder(TrainableModel):
    """Trainable linear autoencoder with Adamax + steplr as in the notebooks."""

    def __init__(self, params: AutoencoderModelConfig, orientation: SnapshotOrientation) -> None:
        """Attach hyperparameters and the snapshot orientation fed to the network."""
        super().__init__(params)
        self._orientation = orientation
        self._module: _LinearAutoencoderModule | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, x_train: NDArray[np.floating]) -> None:
        """Train the autoencoder with MSE reconstruction."""
        torch.manual_seed(self._params.random_seed)
        matrix = np.asarray(x_train, dtype=np.float32)
        tensor = torch.from_numpy(orient(matrix, self._orientation))
        input_dim = int(tensor.shape[1])
        dims = [input_dim, *list(self._params.hidden_layers)]
        self._module = _LinearAutoencoderModule(dims).to(self._device)
        dataset = TensorDataset(tensor)
        loader = DataLoader(
            dataset,
            batch_size=min(self._params.batch_size, len(dataset)),
            shuffle=True,
        )
        optimizer = torch.optim.Adamax(self._module.parameters(), lr=self._params.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self._params.scheduler_step,
            gamma=self._params.scheduler_gamma,
        )
        loss_fn = nn.MSELoss()
        self._module.train()
        for epoch in range(self._params.num_epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self._device)
                optimizer.zero_grad(set_to_none=True)
                recon, _latent = self._module(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            scheduler.step()
            if (epoch + 1) % self._params.log_every_n_epochs == 0 or epoch == 0:
                LOGGER.info(
                    "Autoencoder epoch %s/%s loss=%.6f",
                    epoch + 1,
                    self._params.num_epochs,
                    epoch_loss / max(len(loader), 1),
                )
        LOGGER.info("Finished autoencoder training on device %s", self._device)

    def evaluate(self, x_test: NDArray[np.floating]) -> dict[str, float]:
        """Return MSE and relative Frobenius reconstruction error."""
        if self._module is None:
            msg = "Call fit() before evaluate()"
            raise RuntimeError(msg)
        matrix = np.asarray(x_test, dtype=np.float32)
        tensor = torch.from_numpy(orient(matrix, self._orientation)).to(self._device)
        self._module.eval()
        with torch.no_grad():
            recon, _latent = self._module(tensor)
            mse = float(nn.functional.mse_loss(recon, tensor).item())
            num = float(torch.linalg.norm(tensor - recon).item())
            den = float(torch.linalg.norm(tensor).item())
        relative = num / den if den > 0.0 else float("inf")
        LOGGER.info("Autoencoder evaluation mse=%.6f relative=%.6f", mse, relative)
        return {"mse": mse, "relative_frobenius": relative}
