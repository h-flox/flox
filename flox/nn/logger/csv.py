from pathlib import Path

from flox.nn.logger.base import BaseLogger


class CSVLogger(BaseLogger):
    def to_csv(self, filename: str | Path | None = None) -> str | None:
        """
        Combines all existing records into a CSV-formatted string. If passed a filename,
        writes to that file; otherwise, returns the string.
        """
        return self.to_pandas().to_csv(filename)
