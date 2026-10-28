"""Counting side effects by identity, for the scenario suite's duplicate rate.

A scenario test can assert that a retry "did not do it again" for the one
effect it is about. What it cannot see is an effect it was not about: a
transition written twice while the test was watching the calendar. So every
scenario in `tests/graph_scenarios/test_recovery_scenarios.py` also records
every outward effect it can observe -- from the fake provider, the shared
event log or the database, never from what the system says it did -- into a
`SideEffectLedger`, keyed by what makes two effects *the same* effect:

- a provider write by its idempotency key;
- an application transition by `(application_id, from, to)`;
- a stored communication event by the provider message id;
- a closed follow-up by its checkpoint id.

`duplicate_side_effect_rate` is then `duplicates / total`, where a duplicate
is any observation of an identity beyond its first. The suite's acceptance
criterion is that it stays at exactly zero.

Identities are only comparable within one deployment: two scenarios that each
submit the fixture application to their own fake provider made two
submissions, not one twice. A suite-wide ledger hands each scenario
`ledger.scoped(name)`, which files everything that scenario records under its
own name.
"""

from collections import Counter
from collections.abc import Hashable, Iterable

#: Effect kinds. Constants because the scenarios and the ledger's report both
#: spell them, and a typo would file an effect under a kind nobody counts.
EXTERNAL_WRITE = "external.write"
CALENDAR_CREATE = "calendar.create"
APPLICATION_TRANSITION = "application.transition"
COMMUNICATION_EVENT = "communication_event"
FOLLOW_UP_CLOSED = "follow_up.closed"
FOLLOW_UP_DRAFTED = "follow_up.drafted"


class SideEffectLedger:
    """Observed side effects, counted by `(kind, identity)`."""

    def __init__(self) -> None:
        self._counts: Counter[tuple[str, Hashable]] = Counter()

    def record(self, kind: str, identity: Hashable) -> None:
        """Count one observation of one effect."""
        self._counts[(kind, identity)] += 1

    def record_all(self, kind: str, identities: Iterable[Hashable]) -> None:
        """Count one observation per identity, repeats included."""
        for identity in identities:
            self.record(kind, identity)

    @property
    def total(self) -> int:
        """Every observation, duplicates included."""
        return sum(self._counts.values())

    @property
    def duplicates(self) -> int:
        """Observations beyond the first of each identity."""
        return sum(count - 1 for count in self._counts.values() if count > 1)

    @property
    def duplicate_side_effect_rate(self) -> float:
        """`duplicates / total`; zero for a ledger that saw nothing."""
        return self.duplicates / self.total if self.total else 0.0

    def duplicated(self) -> dict[tuple[str, Hashable], int]:
        """Each identity seen more than once, with its count. For failure messages."""
        return {key: count for key, count in self._counts.items() if count > 1}

    def kinds(self) -> set[str]:
        """The effect kinds anything was recorded under."""
        return {kind for kind, _identity in self._counts}

    def scoped(self, scope: str) -> "ScopedLedger":
        """A view that records into this ledger with every identity under `scope`."""
        return ScopedLedger(self, scope)


class ScopedLedger:
    """One deployment's view of a shared `SideEffectLedger`."""

    def __init__(self, ledger: SideEffectLedger, scope: str):
        self.ledger = ledger
        self.scope = scope

    def record(self, kind: str, identity: Hashable) -> None:
        self.ledger.record(kind, (self.scope, identity))

    def record_all(self, kind: str, identities: Iterable[Hashable]) -> None:
        for identity in identities:
            self.record(kind, identity)


__all__ = [
    "APPLICATION_TRANSITION",
    "CALENDAR_CREATE",
    "COMMUNICATION_EVENT",
    "EXTERNAL_WRITE",
    "FOLLOW_UP_CLOSED",
    "FOLLOW_UP_DRAFTED",
    "ScopedLedger",
    "SideEffectLedger",
]
