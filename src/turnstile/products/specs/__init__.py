"""L2 product specs — one AgentSpec subclass per product; siblings, never
importing each other (enforced by the spec-independence contract).

`SPECS` is the product registry, keyed by AgentSpec.id() (what cfg.spec_id
selects): adding an agent is one spec module plus one line here — root never
changes."""

from turnstile.products.spec import AgentSpec
from turnstile.products.specs.support_bot import SupportBotSpec

SPECS: dict[str, type[AgentSpec]] = {cls().id(): cls for cls in (SupportBotSpec,)}
