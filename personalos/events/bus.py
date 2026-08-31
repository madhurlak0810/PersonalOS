"""Event bus for publishing and subscribing to domain events."""

import logging
from typing import Callable, Dict, List

from personalos.domain.models import Event, EventType

logger = logging.getLogger(__name__)


class EventBus:
    """Simple event bus for domain events."""

    def __init__(self):
        """Initialize event bus."""
        self._subscribers: Dict[EventType, List[Callable]] = {}

    def subscribe(self, event_type: EventType, handler: Callable):
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.info(f"Subscribed to {event_type}")

    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Unsubscribe from an event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(handler)

    async def publish(self, event: Event):
        """Publish an event."""
        logger.info(f"Publishing event: {event.event_type}")
        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                if hasattr(handler, "__await__"):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {str(e)}", exc_info=True)


# Global event bus instance
_event_bus = EventBus()


def get_event_bus() -> EventBus:
    """Get the global event bus."""
    return _event_bus
