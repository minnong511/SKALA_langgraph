"""Reuse expensive resources while invalidating changed models and indexes."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from kv_cache_agent.rag import embeddings as module
from kv_cache_agent.rag.vector_store import (
    build_vector_store,
    clear_vector_store_cache,
    load_vector_store,
    save_vector_store,
    search_vector_store,
)
from tests.test_paper_rag import FakeEmbeddings


def test_embedding_model_load_is_single_flight_and_configuration_aware(monkeypatch):
    constructor = Mock(side_effect=lambda **_: object())
    monkeypatch.setattr(module, "HuggingFaceEmbeddings", constructor)
    module.clear_embedding_cache()
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda _: module.get_embeddings(), range(16)))
    assert all(value is values[0] for value in values)
    assert constructor.call_count == 1
    monkeypatch.setattr(module, "EMBEDDING_MODEL", "another-model")
    assert module.get_embeddings() is not values[0]
    assert constructor.call_count == 2
    module.clear_embedding_cache()


def test_vector_index_reuses_load_and_invalidates_on_save_or_file_change(tmp_path):
    embeddings = FakeEmbeddings()
    save_vector_store(
        build_vector_store([Document(page_content="TurboQuant initial")], embeddings),
        tmp_path,
    )
    original = load_vector_store(tmp_path, embeddings)
    assert load_vector_store(tmp_path, embeddings) is original
    save_vector_store(
        build_vector_store([Document(page_content="CXL updated")], embeddings), tmp_path
    )
    changed = load_vector_store(tmp_path, embeddings)
    assert changed is not original
    assert "updated" in search_vector_store(changed, "CXL", top_k=1)[0][0].page_content
    (tmp_path / "index.pkl").touch()
    assert load_vector_store(tmp_path, embeddings) is not changed
    assert load_vector_store(tmp_path, FakeEmbeddings()) is not changed
    clear_vector_store_cache()


def test_missing_index_does_not_load_large_embedding_model(tmp_path, monkeypatch):
    import kv_cache_agent.rag.vector_store as store

    model = Mock()
    monkeypatch.setattr(store, "get_embeddings", model)
    with pytest.raises(FileNotFoundError):
        load_vector_store(tmp_path)
    model.assert_not_called()
