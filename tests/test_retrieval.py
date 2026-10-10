import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import rrf
from StreamlitApp import split_into_chunks


# ---- RRF ----

def test_rrf_item_in_both_lists_wins():
    # "b" is 2nd in both lists; "a" and "c" are each 1st in only one list.
    # b: 2/(60+2) = 0.03226   a: 1/(60+1) = 0.01639   -> b first
    assert rrf([["a", "b"], ["c", "b"]])[0] == "b"


def test_rrf_exact_scores_decide_order():
    # a: ranks 1 and 3 -> 1/61 + 1/63 = 0.03227
    # b: ranks 2 and 1 -> 1/62 + 1/61 = 0.03252
    # c: ranks 3 and 2 -> 1/63 + 1/62 = 0.03200
    assert rrf([["a", "b", "c"], ["b", "c", "a"]]) == ["b", "a", "c"]


def test_rrf_single_list_keeps_order():
    assert rrf([["x", "y", "z"]]) == ["x", "y", "z"]


def test_rrf_includes_items_from_either_list():
    assert set(rrf([["a"], ["b"]])) == {"a", "b"}


def test_rrf_empty():
    assert rrf([]) == []
    assert rrf([[], []]) == []


# ---- Chunking (size 1000, overlap 200 -> a new chunk starts every 800 chars) ----

def test_short_text_is_one_chunk():
    assert split_into_chunks("hello") == ["hello"]


def test_empty_text_gives_no_chunks():
    assert split_into_chunks("") == []


def test_chunks_are_at_most_chunk_size():
    chunks = split_into_chunks("x" * 2500)
    assert all(len(c) <= 1000 for c in chunks)


def test_chunk_start_positions_and_overlap():
    text = "".join(str(i % 10) for i in range(2500))
    chunks = split_into_chunks(text)
    assert len(chunks) == 4                    # starts at 0, 800, 1600, 2400
    assert chunks[1] == text[800:1800]
    assert chunks[0][-200:] == chunks[1][:200]  # 200-char overlap between neighbours
