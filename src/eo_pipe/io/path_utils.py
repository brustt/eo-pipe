from abc import ABC, abstractmethod
from pathlib import Path


class PathStrategy(ABC):
    """Generates an output file path for a single step execution."""

    @abstractmethod
    def resolve(
        self,
        step_name: str,
        input_path: Path,
        batch_index: int,
        output_dir: Path,
    ) -> Path:
        """Return the output path for one input file.

        Args:
            step_name: Name of the current step.
            input_path: The input file being processed.
            batch_index: Zero-based position within the current batch.
            output_dir: Target directory for outputs.

        Returns:
            Resolved output :class:`Path`.
        """
        ...


class PrefixedPathStrategy(PathStrategy):
    """Default strategy: ``{output_dir}/{step_name}_{input_stem}{ext}``."""

    def resolve(
        self,
        step_name: str,
        input_path: Path,
        batch_index: int,
        output_dir: Path,
    ) -> Path:
        return output_dir / f"{step_name}_{input_path.name}"


class IndexedPathStrategy(PathStrategy):
    """Numbered strategy: ``{output_dir}/{step_name}_{idx:03d}_{input_stem}{ext}``."""

    def resolve(
        self,
        step_name: str,
        input_path: Path,
        batch_index: int,
        output_dir: Path,
    ) -> Path:
        return output_dir / f"{step_name}_{batch_index:03d}_{input_path.name}"


class NamedPathStrategy(PathStrategy):
    """Fixed-name strategy: ``{output_dir}/{step_name}{ext}``.

    Intended for steps that produce a single named output independent of
    any specific input (e.g. merge steps).  The caller passes the desired
    name as the ``step_name`` argument to :meth:`resolve`; ``input_path``
    and ``batch_index`` are ignored.

    Args:
        ext: File extension including the leading dot.  Defaults to ``".tif"``.
    """

    def __init__(self, ext: str = ".tif") -> None:
        self._ext = ext

    def resolve(
        self,
        step_name: str,
        input_path: Path,
        batch_index: int,
        output_dir: Path,
    ) -> Path:
        return output_dir / f"{step_name}{self._ext}"
