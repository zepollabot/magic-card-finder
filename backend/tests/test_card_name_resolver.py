import pytest

from app.services.card_name_resolver import ScryfallCardNameResolver


class FakeScryfallClient:
    def __init__(self) -> None:
        self.named_calls = []
        self.search_by_name_calls = []
        self._named_results = {}
        self._search_results = {}

    def set_named_result(self, name: str, set_code: str | None, result):
        key = (name, set_code)
        self._named_results[key] = result

    def set_search_results(self, name: str, results):
        self._search_results[name] = results

    async def named(self, fuzzy_name: str, set_code: str | None = None):
        self.named_calls.append((fuzzy_name, set_code))
        return self._named_results.get((fuzzy_name, set_code))

    async def search_by_name(self, raw_name: str):
        self.search_by_name_calls.append(raw_name)
        return self._search_results.get(raw_name, [])


@pytest.mark.asyncio
async def test_resolver_uses_first_fuzzy_hit():
    client = FakeScryfallClient()
    card = {"name": "Tarmogoyf", "id": "id-1"}
    client.set_named_result("Tarmogoyf", "SET", card)

    resolver = ScryfallCardNameResolver(client)  # type: ignore[arg-type]

    result = await resolver.resolve("Tarmogoyf", set_hint="SET")

    assert result == [card]
    assert client.named_calls == [("Tarmogoyf", "SET")]
    assert client.search_by_name_calls == []


@pytest.mark.asyncio
async def test_resolver_falls_back_to_fuzzy_without_set():
    client = FakeScryfallClient()
    card = {"name": "Tarmogoyf", "id": "id-2"}
    client.set_named_result("Tarmogoyf", "WrongSet", None)
    client.set_named_result("Tarmogoyf", None, card)

    resolver = ScryfallCardNameResolver(client)  # type: ignore[arg-type]

    result = await resolver.resolve("Tarmogoyf", set_hint="WrongSet")

    assert result == [card]
    assert client.named_calls == [("Tarmogoyf", "WrongSet"), ("Tarmogoyf", None)]
    assert client.search_by_name_calls == []


@pytest.mark.asyncio
async def test_resolver_uses_search_fallback_and_prefers_exact_match():
    client = FakeScryfallClient()
    # No fuzzy results at all
    client.set_named_result("Tarmogoyf", None, None)

    candidates = [
        {"name": "Tarmogoyf, Judge Promo", "id": "id-2"},
        {"name": "Tarmogoyf", "id": "id-1"},
    ]
    client.set_search_results("Tarmogoyf", candidates)

    resolver = ScryfallCardNameResolver(client)  # type: ignore[arg-type]

    result = await resolver.resolve("Tarmogoyf", set_hint=None)

    assert result == candidates
    assert client.search_by_name_calls == ["Tarmogoyf"]


@pytest.mark.asyncio
async def test_resolver_returns_none_when_no_candidates():
    client = FakeScryfallClient()
    client.set_named_result("Unknown", None, None)
    client.set_search_results("Unknown", [])

    resolver = ScryfallCardNameResolver(client)  # type: ignore[arg-type]

    result = await resolver.resolve("Unknown", set_hint=None)

    assert result == []

