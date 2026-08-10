from agentic_memory_nav.common.types import MemoryItem, MemoryType, new_id
from agentic_memory_nav.memory.sqlite_store import SQLiteMemory


def test_sqlite_memory_retrieval_and_provenance(tmp_path):
    memory = SQLiteMemory(tmp_path / "memory.db")
    item = MemoryItem(
        new_id("mem"),
        MemoryType.SEMANTIC,
        "red cup in kitchen",
        {"entity_id": "cup-1"},
        1.0,
        (1.0, 0.0, 2.0),
        0.9,
        ["frame-1"],
    )
    memory.add_observation(item)
    assert memory.retrieve_by_text("red cup")[0].provenance == ["frame-1"]
    assert memory.retrieve_by_region((1.0, 0.0, 2.0), 0.1)[0].memory_id == item.memory_id
    memory.close()
