"""Shared Rich progress display for training and data-processing scripts."""

from collections.abc import Iterable, Mapping

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table


class RichProgress:
    """Iterable progress bar with optional multi-line status rows.

    ``set_postfix()`` keeps the small, tqdm-like one-line API used by the
    simple training scripts. ``set_status()`` adds a table below the bar for
    metrics that are easier to read across multiple lines.
    """

    def __init__(
        self,
        iterable: Iterable | None = None,
        *,
        total: int | None = None,
        description: str | None = None,
        desc: str | None = None,
        unit: str = "it",
        refresh_per_second: int = 10,
        transient: bool = False,
    ):
        if description is not None and desc is not None and description != desc:
            raise ValueError("description and desc must match when both are given")
        if description is None:
            description = desc or ""

        self.iterable = iterable
        if total is None and iterable is not None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[{task.fields[postfix]}]"),
            TimeRemainingColumn(),
        )
        self.task_id = self.progress.add_task(
            description, total=total, postfix="", unit=unit
        )
        self.live = Live(
            self.progress,
            refresh_per_second=refresh_per_second,
            transient=transient,
        )
        self._status: Table | None = None
        self._started = False

    def _renderable(self):
        if self._status is None:
            return self.progress
        return Group(self.progress, self._status)

    def _refresh(self):
        if self._started:
            self.live.update(self._renderable())

    def __enter__(self):
        self.live.start()
        self._started = True
        self._refresh()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __iter__(self):
        if self.iterable is None:
            raise TypeError("RichProgress requires an iterable for iteration")
        with self:
            for item in self.iterable:
                yield item
                self.update(1)

    def update(self, advance=0, **fields):
        if not self._started:
            self.__enter__()
        postfix = fields.pop("postfix", None)
        status = fields.pop("status", None)
        if fields:
            postfix = " ".join(f"{key}={value}" for key, value in fields.items())
        if status is not None:
            postfix = status
        self.progress.update(
            self.task_id,
            advance=advance,
            postfix="" if postfix is None else str(postfix),
        )
        self._refresh()

    def set_postfix(self, **values):
        """Update the compact one-line text following the progress bar."""
        self.update(
            postfix=" ".join(f"{key}={value}" for key, value in values.items())
        )

    def set_status(
        self,
        rows: Mapping[str, object] | None = None,
        **values: object,
    ):
        """Replace the multi-line status table below the progress bar."""
        if rows is not None and values:
            combined = dict(rows)
            combined.update(values)
            rows = combined
        elif rows is None:
            rows = values

        status = Table.grid(padding=(0, 1))
        for label, value in rows.items():
            status.add_row(str(label), str(value))
        self._status = status
        self._refresh()

    def close(self):
        if self._started:
            self.live.stop()
            self._started = False
