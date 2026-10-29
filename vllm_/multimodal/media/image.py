# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO
from pathlib import Path
import threading

import numpy as np
import pybase64
import torch
from PIL import Image
from torchvision.io import ImageReadMode, decode_jpeg

from vllm.utils.serial_utils import tensor2base64
from vllm.logger import init_logger

from ..image import convert_image_mode, rgba_to_rgb
from .base import MediaIO, MediaWithBytes

MAGIC_NUMPY_PREFIX = b"\x93NUMPY"  # https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#format-version-1-0
_TENSOR_JPEG_DECODE_SUPPORTED: bool | None = None
_TENSOR_JPEG_DECODE_LOCK = threading.Lock()
logger = init_logger(__name__)


def is_tensor_jpeg_decode_supported() -> bool:
    """Probe the torchvision JPEG-to-tensor path once per process."""
    global _TENSOR_JPEG_DECODE_SUPPORTED
    if _TENSOR_JPEG_DECODE_SUPPORTED is not None:
        return _TENSOR_JPEG_DECODE_SUPPORTED

    with _TENSOR_JPEG_DECODE_LOCK:
        if _TENSOR_JPEG_DECODE_SUPPORTED is not None:
            return _TENSOR_JPEG_DECODE_SUPPORTED
        try:
            image = Image.new("RGB", (2, 2), (32, 64, 128))
            with BytesIO() as buffer:
                image.save(buffer, format="JPEG")
                encoded = torch.frombuffer(
                    bytearray(buffer.getvalue()), dtype=torch.uint8
                )
            decoded = decode_jpeg(encoded, mode=ImageReadMode.RGB)
            if decoded.dtype != torch.uint8 or tuple(decoded.shape) != (3, 2, 2):
                raise RuntimeError(
                    "unexpected decoded JPEG tensor "
                    f"dtype={decoded.dtype}, shape={tuple(decoded.shape)}"
                )
        except Exception as exc:
            _TENSOR_JPEG_DECODE_SUPPORTED = False
            logger.warning(
                "Disabling tensor JPEG media decode; torchvision probe failed: %s",
                exc,
            )
        else:
            _TENSOR_JPEG_DECODE_SUPPORTED = True
    return _TENSOR_JPEG_DECODE_SUPPORTED


def disable_tensor_jpeg_decode(reason: object) -> None:
    """Disable tensor JPEG decoding after a runtime codec failure."""
    global _TENSOR_JPEG_DECODE_SUPPORTED
    with _TENSOR_JPEG_DECODE_LOCK:
        was_enabled = _TENSOR_JPEG_DECODE_SUPPORTED is not False
        _TENSOR_JPEG_DECODE_SUPPORTED = False
    if was_enabled:
        logger.warning(
            "Disabling tensor JPEG media decode after runtime failure; "
            "falling back to PIL: %s",
            reason,
        )


class ImageMediaIO(MediaIO[Image.Image | torch.Tensor]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    def __init__(
        self,
        image_mode: str = "RGB",
        decode_to_tensor: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.image_mode = image_mode
        self.decode_to_tensor = decode_to_tensor
        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality, merged with
        # per-request runtime media_io_kwargs via merge_kwargs().
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        self.kwargs = kwargs

        # Extract RGBA background color from kwargs if provided
        # Default to white background for backward compatibility
        rgba_bg = kwargs.get("rgba_background_color", (255, 255, 255))
        # Convert list to tuple for consistency
        if isinstance(rgba_bg, list):
            rgba_bg = tuple(rgba_bg)

        # Validate rgba_background_color format
        if not (
            isinstance(rgba_bg, tuple)
            and len(rgba_bg) == 3
            and all(isinstance(c, int) and 0 <= c <= 255 for c in rgba_bg)
        ):
            raise ValueError(
                "rgba_background_color must be a list or tuple of 3 integers "
                "in the range [0, 255]."
            )
        self.rgba_background_color = rgba_bg

    def _decode_tensor_if_supported(self, data: bytes) -> torch.Tensor | None:
        if (
            not self.decode_to_tensor
            or self.image_mode != "RGB"
            or not is_tensor_jpeg_decode_supported()
        ):
            return None
        if not data.startswith(b"\xff\xd8\xff"):
            return None

        try:
            encoded = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            return decode_jpeg(encoded, mode=ImageReadMode.RGB)
        except Exception as exc:
            disable_tensor_jpeg_decode(exc)
            return None

    def _convert_image_mode(
        self, image: Image.Image | MediaWithBytes[Image.Image]
    ) -> Image.Image:
        """Convert image mode with custom background color."""
        if isinstance(image, MediaWithBytes):
            image = image.media
        if image.mode == self.image_mode:
            return image
        elif image.mode == "RGBA" and self.image_mode == "RGB":
            return rgba_to_rgb(image, self.rgba_background_color)
        else:
            return convert_image_mode(image, self.image_mode)

    def load_bytes(
        self, data: bytes
    ) -> MediaWithBytes[Image.Image | torch.Tensor]:
        tensor = self._decode_tensor_if_supported(data)
        if tensor is not None:
            return MediaWithBytes(tensor, data)
        image = Image.open(BytesIO(data))
        return MediaWithBytes(self._convert_image_mode(image), data)

    def load_base64(
        self, media_type: str, data: str
    ) -> MediaWithBytes[Image.Image | torch.Tensor]:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(
        self, filepath: Path
    ) -> MediaWithBytes[Image.Image | torch.Tensor]:
        with open(filepath, "rb") as f:
            data = f.read()
        tensor = self._decode_tensor_if_supported(data)
        if tensor is not None:
            return MediaWithBytes(tensor, data)
        image = Image.open(BytesIO(data))
        return MediaWithBytes(self._convert_image_mode(image), data)

    def encode_base64(
        self,
        media: Image.Image,
        *,
        image_format: str = "PNG",
    ) -> str:
        image = media

        with BytesIO() as buffer:
            image = self._convert_image_mode(image)
            image.save(buffer, image_format)
            data = buffer.getvalue()

        return pybase64.b64encode(data).decode("utf-8")


class ImageEmbeddingMediaIO(MediaIO[torch.Tensor]):
    """Image embedding MediaIO implementation.

    Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    def __init__(self) -> None:
        super().__init__()

    def _load_pickled_torch(self, data: bytes) -> torch.Tensor:
        buffer = BytesIO(data)
        # Enable sparse tensor integrity checks to prevent out-of-bounds
        # writes from maliciously crafted tensors
        with torch.sparse.check_sparse_tensor_invariants():
            tensor = torch.load(buffer, weights_only=True)
            return tensor.to_dense()

    def _load_numpy(self, data: bytes) -> torch.Tensor:
        with BytesIO(data) as buffer:
            return torch.from_numpy(np.load(buffer))

    def load_bytes(self, data: bytes) -> torch.Tensor:
        if data[:6] == MAGIC_NUMPY_PREFIX:
            return self._load_numpy(data)

        return self._load_pickled_torch(data)

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> torch.Tensor:
        if filepath.suffix == ".npy":
            return torch.from_numpy(np.load(filepath))

        with torch.sparse.check_sparse_tensor_invariants():
            tensor = torch.load(filepath, weights_only=True)
            return tensor.to_dense()

    def encode_base64(self, media: torch.Tensor) -> str:
        return tensor2base64(media)
