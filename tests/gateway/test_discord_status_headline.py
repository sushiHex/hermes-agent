from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord.status_headline import (
    QuotaSummary,
    StatusHeadline,
    StatusSnapshot,
    collect_status_snapshot,
)


class FakeRole:
    def __init__(self, resource_id: int) -> None:
        self.id = resource_id


class FakeCategory:
    def __init__(self, resource_id: int, overwrites: tuple) -> None:
        self.id = resource_id
        self.name = "STATUS"
        self.overwrites = overwrites
        self.voice_channels: list[FakeVoice] = []

    @property
    def channels(self):
        return list(self.voice_channels)


class FakeVoice:
    def __init__(
        self,
        resource_id: int,
        name: str,
        *,
        category_id: int,
        position: int,
        overwrites: tuple,
    ) -> None:
        self.id = resource_id
        self.name = name
        self.category_id = category_id
        self.position = position
        self.overwrites = overwrites
        self.edits: list[dict] = []
        self.deleted = False

    async def edit(self, **changes):
        self.edits.append(dict(changes))
        for key, value in changes.items():
            setattr(self, key, value)
        return self

    async def delete(self, *, reason: str):
        del reason
        self.deleted = True


class FakeDiscord:
    CategoryChannel = FakeCategory
    Role = FakeRole
    VoiceChannel = FakeVoice


class FakeHTTP:
    def __init__(self, guild) -> None:
        self.guild = guild

    async def bulk_channel_update(self, guild_id, payload, *, reason):
        del reason
        assert guild_id == self.guild.id
        self.guild.bulk_updates.append([dict(item) for item in payload])
        for item in payload:
            self.guild.get_channel(item["id"]).position = item["position"]


class FakeGuild:
    def __init__(self) -> None:
        self.id = 10
        self.overwrites = (("everyone", False, False), ("helper", True, False))
        self.role = FakeRole(30)
        self.category = FakeCategory(20, self.overwrites)
        names = (
            (40, "Model: GPT-5.6 Sol"),
            (41, "Effort: High"),
            (42, "ChatGPT: unavailable"),
            (43, "Claude: unavailable"),
        )
        self.category.voice_channels = [
            FakeVoice(
                resource_id,
                name,
                category_id=20,
                position=position,
                overwrites=self.overwrites,
            )
            for position, (resource_id, name) in enumerate(names, 10)
        ]
        self.me = SimpleNamespace(roles=[self.role])
        self.created: list[FakeVoice] = []
        self.bulk_updates: list[list[dict]] = []
        self.fetch_count = 0
        self._state = SimpleNamespace(http=FakeHTTP(self))

    def get_role(self, resource_id: int):
        return self.role if resource_id == self.role.id else None

    def get_channel(self, resource_id: int):
        if resource_id == self.category.id:
            return self.category
        return next(
            (row for row in self.category.voice_channels if row.id == resource_id),
            None,
        )

    async def fetch_channels(self):
        self.fetch_count += 1
        return [self.category, *self.category.voice_channels]

    async def create_voice_channel(
        self,
        name: str,
        *,
        category,
        overwrites,
        position: int,
        reason: str,
    ):
        del reason
        row = FakeVoice(
            44,
            name,
            category_id=category.id,
            position=position,
            overwrites=overwrites,
        )
        index = self.category.voice_channels.index(self.get_channel(40))
        self.category.voice_channels.insert(index, row)
        self.created.append(row)
        return row


FOUR_ROW_RECORD = {
    "version": 1,
    "guild_id": 10,
    "helper_role_id": 30,
    "category_id": 20,
    "channel_ids": {
        "model": 40,
        "effort": 41,
        "chatgpt": 42,
        "claude": 43,
    },
}


async def snapshot() -> StatusSnapshot:
    return StatusSnapshot(
        hermes_version="0.21.0",
        model="GPT-5.6 Sol",
        effort="High",
        chatgpt=QuotaSummary(False),
        claude=QuotaSummary(False),
    )


def test_snapshot_uses_current_gateway_model_context(monkeypatch) -> None:
    import agent.account_usage as account_usage
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model_context",
        lambda: SimpleNamespace(model="gpt-5.6-sol"),
    )
    monkeypatch.setattr(account_usage, "fetch_account_usage", lambda provider: None)
    runner = SimpleNamespace(
        _load_reasoning_config=lambda model: {"enabled": True, "effort": "high"}
    )

    result = asyncio.run(collect_status_snapshot(runner))

    assert result.model == "GPT-5.6 Sol"
    assert result.effort == "High"


def test_default_ownership_path_is_shared_fleet_root(monkeypatch, tmp_path: Path) -> None:
    import hermes_constants

    fleet_root = tmp_path / "fleet"
    profile_root = tmp_path / "profile"
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: fleet_root)
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: profile_root)

    headline = StatusHeadline(
        client=None,
        discord_module=FakeDiscord,
        config={"enabled": True, "guild_id": 10, "category_name": "STATUS"},
        runner=None,
        snapshot_provider=snapshot,
    )

    assert headline.store.path == fleet_root / "state" / "discord-status-sidebar" / "10.json"


def test_exact_four_row_surface_adds_only_hermes_and_restarts_noop(
    tmp_path: Path,
) -> None:
    asyncio.run(_exercise_four_row_migration(tmp_path))


async def _exercise_four_row_migration(tmp_path: Path) -> None:
    ownership = tmp_path / "10.json"
    ownership.write_text(
        json.dumps(FOUR_ROW_RECORD, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    guild = FakeGuild()
    for row in guild.category.voice_channels:
        row.name = f"stale-{row.id}"
    create_voice_channel = guild.create_voice_channel

    async def create_reordered_voice_channel(*args, **kwargs):
        row = await create_voice_channel(*args, **kwargs)
        row.name = "discord-normalized"
        row.position = 99
        guild.category.voice_channels.remove(row)
        guild.category.voice_channels.append(row)
        return row

    guild.create_voice_channel = create_reordered_voice_channel
    client = SimpleNamespace(get_guild=lambda guild_id: guild if guild_id == 10 else None)

    headline = StatusHeadline(
        client,
        FakeDiscord,
        {"enabled": True, "guild_id": 10, "category_name": "STATUS"},
        runner=None,
        ownership_path=ownership,
        snapshot_provider=snapshot,
    )
    await headline.run_once()

    assert [row.id for row in guild.created] == [44]
    migrated = json.loads(ownership.read_text(encoding="utf-8"))
    assert migrated == {
        **FOUR_ROW_RECORD,
        "channel_ids": {**FOUR_ROW_RECORD["channel_ids"], "hermes": 44},
    }
    expected_names = {
        44: "Hermes v0.21.0",
        40: "Model: GPT-5.6 Sol",
        41: "Effort: High",
        42: "ChatGPT: unavailable",
        43: "Claude: unavailable",
    }
    assert {
        row.id: row.name for row in guild.category.voice_channels
    } == expected_names
    assert {
        row.id: row.position for row in guild.category.voice_channels
    } == {44: 10, 40: 11, 41: 12, 42: 13, 43: 14}
    assert guild.bulk_updates == [
        [
            {"id": 44, "position": 10},
            {"id": 40, "position": 11},
            {"id": 41, "position": 12},
            {"id": 42, "position": 13},
            {"id": 43, "position": 14},
        ]
    ]
    assert guild.fetch_count == 3
    assert all(
        "position" not in edit
        for row in guild.category.voice_channels
        for edit in row.edits
    )
    edit_counts = {
        row.id: len(row.edits) for row in guild.category.voice_channels
    }
    bulk_count = len(guild.bulk_updates)
    fetch_count = guild.fetch_count

    await headline.run_once()
    restarted = StatusHeadline(
        client,
        FakeDiscord,
        {"enabled": True, "guild_id": 10, "category_name": "STATUS"},
        runner=None,
        ownership_path=ownership,
        snapshot_provider=snapshot,
    )
    await restarted.run_once()

    assert [row.id for row in guild.created] == [44]
    assert {
        row.id: len(row.edits) for row in guild.category.voice_channels
    } == edit_counts
    assert len(guild.bulk_updates) == bulk_count
    assert guild.fetch_count == fetch_count + 2


def test_adapter_owns_exactly_one_headline_task_and_cancels_it() -> None:
    asyncio.run(_exercise_adapter_task_lifecycle())


def test_fresh_topology_drift_refuses_before_mutation(tmp_path: Path) -> None:
    asyncio.run(_exercise_fresh_topology_refusal(tmp_path))


async def _exercise_fresh_topology_refusal(tmp_path: Path) -> None:
    ownership = tmp_path / "10.json"
    ownership.write_text(
        json.dumps(FOUR_ROW_RECORD, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    guild = FakeGuild()
    foreign = FakeVoice(
        99,
        "foreign",
        category_id=20,
        position=14,
        overwrites=guild.overwrites,
    )

    async def fresh_channels():
        guild.fetch_count += 1
        return [guild.category, *guild.category.voice_channels, foreign]

    guild.fetch_channels = fresh_channels
    client = SimpleNamespace(get_guild=lambda guild_id: guild if guild_id == 10 else None)
    headline = StatusHeadline(
        client,
        FakeDiscord,
        {"enabled": True, "guild_id": 10, "category_name": "STATUS"},
        runner=None,
        ownership_path=ownership,
        snapshot_provider=snapshot,
    )

    with pytest.raises(RuntimeError, match="topology drifted"):
        await headline.run_once()

    assert guild.created == []
    assert guild.bulk_updates == []
    assert all(not row.edits for row in guild.category.voice_channels)


async def _exercise_adapter_task_lifecycle() -> None:
    adapter = object.__new__(DiscordAdapter)
    adapter.config = PlatformConfig(
        enabled=True,
        token="not-used",
        extra={"status_sidebar": {"enabled": True, "guild_id": 10}},
    )
    adapter._client = object()
    adapter._status_headline_task = None
    started = asyncio.Event()

    async def run_forever():
        started.set()
        await asyncio.Event().wait()

    adapter._run_status_headline = run_forever
    first = adapter._ensure_status_headline_task()
    assert first is adapter._ensure_status_headline_task()
    await started.wait()

    await adapter._cancel_status_headline_task()

    assert first.done() and first.cancelled()
    assert adapter._status_headline_task is None


def test_invalid_headline_config_stays_noncritical() -> None:
    asyncio.run(_exercise_invalid_config_containment())


async def _exercise_invalid_config_containment() -> None:
    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="not-used",
            extra={"status_sidebar": {"enabled": True, "guild_id": "invalid"}},
        )
    )
    adapter._client = object()
    adapter.gateway_runner = None

    task = asyncio.create_task(adapter._run_status_headline())
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_headline_task_holds_guild_lock_until_cancel(monkeypatch) -> None:
    asyncio.run(_exercise_guild_lock(monkeypatch))


async def _exercise_guild_lock(monkeypatch) -> None:
    import gateway.status as gateway_status
    import plugins.platforms.discord.status_headline as headline_module

    acquired: list[tuple] = []
    released: list[tuple] = []
    started = asyncio.Event()
    monkeypatch.setattr(
        gateway_status,
        "acquire_scoped_lock",
        lambda scope, identity, metadata=None: (
            acquired.append((scope, identity, metadata)) or True,
            None,
        ),
    )
    monkeypatch.setattr(
        gateway_status,
        "release_scoped_lock",
        lambda scope, identity: released.append((scope, identity)),
    )

    class BlockingHeadline:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.guild_id = 10

        async def run_once(self):
            started.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(headline_module, "StatusHeadline", BlockingHeadline)
    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="not-used",
            extra={"status_sidebar": {"enabled": True, "guild_id": 10}},
        )
    )
    adapter._client = object()
    adapter.gateway_runner = None

    task = asyncio.create_task(adapter._run_status_headline())
    await started.wait()
    assert acquired == [
        ("discord-status-sidebar-guild", "10", {"guild_id": 10})
    ]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released == [("discord-status-sidebar-guild", "10")]


def test_same_process_adapters_cannot_share_headline_guild(monkeypatch) -> None:
    asyncio.run(_exercise_same_process_guild_exclusion(monkeypatch))


async def _exercise_same_process_guild_exclusion(monkeypatch) -> None:
    import gateway.status as gateway_status
    import plugins.platforms.discord.status_headline as headline_module

    acquired: list[str] = []
    released: list[str] = []
    entered: list[object] = []
    first_started = asyncio.Event()

    monkeypatch.setattr(
        gateway_status,
        "acquire_scoped_lock",
        lambda scope, identity, metadata=None: (
            acquired.append(identity) or True,
            {"pid": "same-process"},
        ),
    )
    monkeypatch.setattr(
        gateway_status,
        "release_scoped_lock",
        lambda scope, identity: released.append(identity),
    )

    class BlockingHeadline:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.guild_id = 10

        async def run_once(self):
            entered.append(self)
            first_started.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(headline_module, "StatusHeadline", BlockingHeadline)

    def adapter() -> DiscordAdapter:
        value = DiscordAdapter(
            PlatformConfig(
                enabled=True,
                token="not-used",
                extra={"status_sidebar": {"enabled": True, "guild_id": 10}},
            )
        )
        value._client = object()
        value.gateway_runner = None
        return value

    owner_task = asyncio.create_task(adapter()._run_status_headline())
    await first_started.wait()
    rejected_task = asyncio.create_task(adapter()._run_status_headline())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert len(entered) == 1
    assert acquired == ["10"]

    rejected_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await rejected_task
    assert released == []

    owner_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner_task
    assert released == ["10"]


def test_headline_uses_established_refresh_interval_key(monkeypatch) -> None:
    asyncio.run(_exercise_refresh_interval(monkeypatch))


async def _exercise_refresh_interval(monkeypatch) -> None:
    import gateway.status as gateway_status
    import plugins.platforms.discord.adapter as adapter_module
    import plugins.platforms.discord.status_headline as headline_module

    delays: list[float] = []

    class OneShotHeadline:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.guild_id = 10

        async def run_once(self):
            return None

    async def stop_after_delay(delay: float):
        delays.append(delay)
        raise asyncio.CancelledError

    monkeypatch.setattr(headline_module, "StatusHeadline", OneShotHeadline)
    monkeypatch.setattr(adapter_module.asyncio, "sleep", stop_after_delay)
    monkeypatch.setattr(
        gateway_status,
        "acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(gateway_status, "release_scoped_lock", lambda scope, identity: None)
    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="not-used",
            extra={
                "status_sidebar": {
                    "enabled": True,
                    "guild_id": 10,
                    "refresh_interval_seconds": 600,
                }
            },
        )
    )
    adapter._client = object()
    adapter.gateway_runner = None

    with pytest.raises(asyncio.CancelledError):
        await adapter._run_status_headline()

    assert delays == [600.0]
