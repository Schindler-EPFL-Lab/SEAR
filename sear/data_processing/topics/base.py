from datetime import datetime


class BaseTopic:
    """Represents a base class for topics with a timestamp."""

    def __init__(self, timestamp: datetime) -> None:
        """Initializes the BaseTopic with `timestamp` when the topic was measured."""
        self._timestamp = timestamp

    @property
    def timestamp(self) -> datetime:
        """Returns the timestamp when the topic was measured."""
        return self._timestamp
