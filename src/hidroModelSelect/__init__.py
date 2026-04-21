import importlib.metadata
from .distCompare import HidroModelSelector


try:
    # Lee la versión desde los metadatos del paquete instalado (definido en pyproject.toml)
    __version__ = importlib.metadata.version("hidroModelSelect")
except importlib.metadata.PackageNotFoundError:
    # Fallback por si el paquete no está instalado (ej. durante el desarrollo)
    __version__ = "0.0.0-dev"

__all__ = ["HidroModelSelector"]
