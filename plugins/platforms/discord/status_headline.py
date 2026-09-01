from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

from utils import atomic_json_write

logger = logging.getLogger(__name__)

_LEGACY_ROWS = ("model", "effort", "chatgpt", "claude")
_ALL_ROWS = ("hermes", *_LEGACY_ROWS)
_RECORD_KEYS = {
    "version",
    "guild_id",
    "helper_role_id",
    "category_id",
    "channel_ids",
}


@dataclass(frozen=True)
class QuotaSummary:
    available: bool
    remaining_percent: int | None = None
    reset_at: datetime | None = None


@dataclass(frozen=True)
class StatusSnapshot:
    hermes_version: str
    model: str
    effort: str
    chatgpt: QuotaSummary
    claude: QuotaSummary


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"Discord status headline has invalid {label}")
    return value


def _display_model(value: str) -> str:
    text = str(value or "").strip()
    match = re.fullmatch(
        r"(?:[^/]+/)?gpt-(\d+(?:\.\d+)*)(?:-([a-z0-9]+))?",
        text,
        re.IGNORECASE,
    )
    if not match:
        return text or "Not configured"
    version, suffix = match.groups()
    return f"GPT-{version}" + (f" {suffix.title()}" if suffix else "")


def _quota_summary(snapshot: Any, labels: tuple[str, ...]) -> QuotaSummary:
    if snapshot is None:
        return QuotaSummary(False)
    preferred = {label.casefold() for label in labels}
    window = next(
        (
            item
            for item in (getattr(snapshot, "windows", ()) or ())
            if str(getattr(item, "label", "")).casefold() in preferred
        ),
        None,
    )
    used = getattr(window, "used_percent", None) if window is not None else None
    if not isinstance(used, (int, float)) or isinstance(used, bool) or not math.isfinite(used):
        return QuotaSummary(False)
    remaining = max(0, min(100, round(100 - float(used))))
    reset_at = getattr(window, "reset_at", None)
    return QuotaSummary(
        True,
        remaining,
        reset_at if isinstance(reset_at, datetime) else None,
    )


async def collect_status_snapshot(runner: Any) -> StatusSnapshot:
    from agent.account_usage import fetch_account_usage
    from gateway.run import _resolve_gateway_model_context
    from hermes_cli import __version__

    model_context = await asyncio.to_thread(_resolve_gateway_model_context)
    model = model_context.model
    reasoning = None
    resolver = getattr(runner, "_load_reasoning_config", None)
    if callable(resolver):
        try:
            reasoning = resolver(model)
        except Exception:
            logger.debug("Discord status headline effort lookup failed", exc_info=True)
    if reasoning and reasoning.get("enabled") is False:
        effort = "Off"
    elif reasoning and reasoning.get("effort"):
        effort = str(reasoning["effort"]).strip().title()
    else:
        effort = "Default"

    chatgpt_raw, claude_raw = await asyncio.gather(
        asyncio.to_thread(fetch_account_usage, "openai-codex"),
        asyncio.to_thread(fetch_account_usage, "anthropic"),
    )
    return StatusSnapshot(
        hermes_version=str(__version__),
        model=_display_model(model),
        effort=effort,
        chatgpt=_quota_summary(chatgpt_raw, ("Weekly",)),
        claude=_quota_summary(
            claude_raw,
            ("Current week", "Seven-day", "7-day"),
        ),
    )


def _reset_text(reset_at: datetime | None, now: datetime) -> str:
    if reset_at is None:
        return ""
    if reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=timezone.utc)
    seconds = max(0, int((reset_at - now).total_seconds()))
    hours = math.ceil(seconds / 3600)
    if hours >= 24:
        days, remainder = divmod(hours, 24)
        return f" · reset {days}d {remainder}h"
    return f" · reset {hours}h"


def _quota_name(label: str, quota: QuotaSummary, now: datetime) -> str:
    if not quota.available or quota.remaining_percent is None:
        return f"{label}: unavailable"
    return (
        f"{label}: {quota.remaining_percent}%"
        f"{_reset_text(quota.reset_at, now)}"
    )[:100]


def channel_names(snapshot: StatusSnapshot, now: datetime | None = None) -> dict[str, str]:
    now = now or datetime.now(timezone.utc)
    return {
        "hermes": f"Hermes v{snapshot.hermes_version}"[:100],
        "model": f"Model: {snapshot.model}"[:100],
        "effort": f"Effort: {snapshot.effort}"[:100],
        "chatgpt": _quota_name("ChatGPT", snapshot.chatgpt, now),
        "claude": _quota_name("Claude", snapshot.claude, now),
    }


class OwnershipStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Discord status headline requires existing schema-1 ownership"
            ) from exc
        except Exception as exc:
            raise RuntimeError("Discord status headline ownership is unreadable") from exc
        if not isinstance(value, dict):
            raise RuntimeError("Discord status headline ownership is malformed")
        return value

    def save(self, value: Mapping[str, Any]) -> None:
        atomic_json_write(self.path, dict(value), sort_keys=True)


class StatusHeadline:
    """Maintain five rows on one pre-existing, exact-ID Discord surface."""

    def __init__(
        self,
        client: Any,
        discord_module: Any,
        config: Mapping[str, Any],
        runner: Any,
        *,
        ownership_path: Path | None = None,
        snapshot_provider: Callable[[], Awaitable[StatusSnapshot]] | None = None,
    ) -> None:
        self.client = client
        self.discord = discord_module
        self.config = dict(config)
        self.runner = runner
        self.guild_id = _positive_int(self.config.get("guild_id"), "configured guild id")
        self.category_name = str(
            self.config.get("category_name") or "HERMES STATUS"
        ).strip()
        if not self.category_name:
            raise RuntimeError("Discord status headline category name is empty")
        if ownership_path is None:
            from hermes_constants import get_default_hermes_root

            ownership_path = (
                get_default_hermes_root()
                / "state"
                / "discord-status-sidebar"
                / f"{self.guild_id}.json"
            )
        self.store = OwnershipStore(ownership_path)
        self.snapshot_provider = snapshot_provider or (
            lambda: collect_status_snapshot(self.runner)
        )

    def _record(self) -> dict[str, Any]:
        record = self.store.load()
        if set(record) != _RECORD_KEYS or record.get("version") != 1:
            raise RuntimeError("Discord status headline requires exact schema-1 ownership")
        if record.get("guild_id") != self.guild_id:
            raise RuntimeError("Discord status headline ownership guild differs from config")
        role_id = _positive_int(record.get("helper_role_id"), "helper role id")
        category_id = _positive_int(record.get("category_id"), "category id")
        channel_ids = record.get("channel_ids")
        if not isinstance(channel_ids, dict) or frozenset(channel_ids) not in {
            frozenset(_LEGACY_ROWS),
            frozenset(_ALL_ROWS),
        }:
            raise RuntimeError(
                "Discord status headline ownership is not exact four- or five-row state"
            )
        ids = [self.guild_id, role_id, category_id]
        ids.extend(_positive_int(value, f"{key} row id") for key, value in channel_ids.items())
        if len(ids) != len(set(ids)):
            raise RuntimeError("Discord status headline ownership reuses an id")
        return {**record, "channel_ids": dict(channel_ids)}

    def _surface(
        self,
        record: Mapping[str, Any],
        *,
        inventory: Optional[Sequence[Any]] = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        guild = self.client.get_guild(self.guild_id)
        if guild is None:
            raise RuntimeError("Discord status headline guild is unavailable")
        by_id = (
            {getattr(item, "id", None): item for item in inventory}
            if inventory is not None
            else None
        )
        category = (
            by_id.get(record["category_id"])
            if by_id is not None
            else guild.get_channel(record["category_id"])
        )
        role = guild.get_role(record["helper_role_id"])
        if not isinstance(category, self.discord.CategoryChannel):
            raise RuntimeError("Discord status headline owned category is unavailable")
        if not isinstance(role, self.discord.Role):
            raise RuntimeError("Discord status headline owned helper role is unavailable")
        if str(getattr(category, "name", "")) != self.category_name:
            raise RuntimeError("Discord status headline owned category name drifted")
        member_roles = {
            getattr(item, "id", None) for item in getattr(getattr(guild, "me", None), "roles", ())
        }
        if role.id not in member_roles:
            raise RuntimeError("Discord status headline helper role assignment drifted")

        rows: dict[str, Any] = {}
        for key, row_id in record["channel_ids"].items():
            row = by_id.get(row_id) if by_id is not None else guild.get_channel(row_id)
            if not isinstance(row, self.discord.VoiceChannel):
                raise RuntimeError(f"Discord status headline owned {key} row is unavailable")
            if getattr(row, "category_id", None) != category.id:
                raise RuntimeError(f"Discord status headline owned {key} row parent drifted")
            if getattr(row, "overwrites", None) != getattr(category, "overwrites", None):
                raise RuntimeError(f"Discord status headline owned {key} row overwrites drifted")
            rows[key] = row

        child_ids = (
            {
                getattr(item, "id", None)
                for item in inventory
                if getattr(item, "category_id", None) == category.id
            }
            if inventory is not None
            else {
                getattr(item, "id", None)
                for item in getattr(category, "channels", ())
            }
        )
        if child_ids != set(record["channel_ids"].values()):
            raise RuntimeError("Discord status headline owned category topology drifted")
        return guild, category, rows

    async def _migrate(
        self,
        guild: Any,
        category: Any,
        rows: Mapping[str, Any],
        record: dict[str, Any],
        names: Mapping[str, str],
    ) -> None:
        model_position = getattr(rows["model"], "position", None)
        if type(model_position) is not int or model_position < 0:
            raise RuntimeError("Discord status headline Model row position is malformed")

        pending = {**record, "pending": {"kind": "row:hermes"}}
        self.store.save(pending)
        created = await guild.create_voice_channel(
            names["hermes"],
            category=category,
            overwrites=category.overwrites,
            position=model_position,
            reason="Add Hermes version to existing status headline",
        )
        created_id = _positive_int(getattr(created, "id", None), "created Hermes row id")
        if created_id in {
            self.guild_id,
            record["helper_role_id"],
            record["category_id"],
            *record["channel_ids"].values(),
        }:
            raise RuntimeError("Discord status headline created row reuses an id")
        migrated = {
            **record,
            "channel_ids": {**record["channel_ids"], "hermes": created_id},
        }
        self.store.save(migrated)

    @staticmethod
    def _ordered_row_ids(rows: Mapping[str, Any]) -> list[int]:
        positioned = []
        for row in rows.values():
            position = getattr(row, "position", None)
            row_id = getattr(row, "id", None)
            if type(position) is not int or position < 0 or type(row_id) is not int:
                raise RuntimeError("Discord status headline row position is malformed")
            positioned.append((position, row_id))
        return [row_id for _position, row_id in sorted(positioned)]

    async def _refresh(
        self,
        guild: Any,
        record: Mapping[str, Any],
        rows: Mapping[str, Any],
        names: Mapping[str, str],
    ) -> None:
        changed = False
        for key in _ALL_ROWS:
            row = rows[key]
            if str(getattr(row, "name", "")) != names[key]:
                await row.edit(
                    name=names[key],
                    reason="Refresh Hermes status headline",
                )
                changed = True

        desired_ids = [rows[key].id for key in _ALL_ROWS]
        if self._ordered_row_ids(rows) != desired_ids:
            base = min(row.position for row in rows.values())
            payload = [
                {"id": rows[key].id, "position": base + offset}
                for offset, key in enumerate(_ALL_ROWS)
            ]
            await guild._state.http.bulk_channel_update(
                guild.id,
                payload,
                reason="Order Hermes status headline",
            )
            changed = True

        if not changed:
            return

        inventory = await guild.fetch_channels()
        _guild, _category, fresh_rows = self._surface(record, inventory=inventory)
        if any(str(getattr(fresh_rows[key], "name", "")) != names[key] for key in _ALL_ROWS):
            raise RuntimeError("Discord status headline name refresh did not converge")
        if self._ordered_row_ids(fresh_rows) != [fresh_rows[key].id for key in _ALL_ROWS]:
            raise RuntimeError("Discord status headline order refresh did not converge")

    async def run_once(self) -> None:
        if self.config.get("enabled") is not True:
            return
        record = self._record()
        guild = self.client.get_guild(self.guild_id)
        if guild is None:
            raise RuntimeError("Discord status headline guild is unavailable")
        inventory = await guild.fetch_channels()
        guild, category, rows = self._surface(record, inventory=inventory)
        snapshot = await self.snapshot_provider()
        names = channel_names(snapshot)
        if frozenset(record["channel_ids"]) == frozenset(_LEGACY_ROWS):
            if self._ordered_row_ids(rows) != [rows[key].id for key in _LEGACY_ROWS]:
                raise RuntimeError("Discord status headline legacy row order drifted")
            await self._migrate(guild, category, rows, record, names)
            record = self._record()
            inventory = await guild.fetch_channels()
            _guild, _category, rows = self._surface(record, inventory=inventory)
        await self._refresh(guild, record, rows, names)
