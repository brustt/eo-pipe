from typing import Dict, Type

from .base import StepBase


class StepRegistry:
    """Open-for-extension registry mapping step names to :class:`StepBase` classes.

    Built-in steps self-register via the ``@StepRegistry.register`` decorator.
    User-defined steps register identically without modifying any library file.

    Example::

        @StepRegistry.register
        class MyStep(StepBase):
            name = "my_step"
            ...

        step = StepRegistry.create("my_step", threshold=5)
    """

    _registry: Dict[str, Type[StepBase]] = {}

    @classmethod
    def register(cls, step_class: Type[StepBase]) -> Type[StepBase]:
        """Register a step class by its ``name`` attribute.

        Can be used as a decorator or called directly.

        Args:
            step_class: A concrete :class:`StepBase` subclass with a ``name``
                        class variable.

        Returns:
            The unchanged *step_class* (allows decorator chaining).

        Raises:
            TypeError: If *step_class* does not have a ``name`` attribute.
        """
        if not hasattr(step_class, "name"):
            raise TypeError(
                f"{step_class.__name__} must define a 'name' class variable"
            )
        cls._registry[step_class.name] = step_class
        return step_class

    @classmethod
    def create(cls, name: str, **constructor_params) -> StepBase:
        """Instantiate a registered step with optional constructor parameters.

        Args:
            name: Registered step name.
            **constructor_params: Passed to the step's ``__init__``.

        Returns:
            A new step instance.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Step '{name}' is not registered. "
                f"Available steps: {available}"
            )
        return cls._registry[name](**constructor_params)

    @classmethod
    def available(cls) -> list:
        """Return a sorted list of all registered step names."""
        return sorted(cls._registry.keys())
