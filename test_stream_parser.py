from pgvector_rag_agent import ThinkTagStreamParser


def collect_chunks(chunks):
    parser = ThinkTagStreamParser()
    content_parts = []
    reasoning_parts = []

    for chunk in chunks:
        result = parser.feed(chunk)
        content_parts.append(result["content"])
        reasoning_parts.append(result["reasoning"])

    tail = parser.flush()
    content_parts.append(tail["content"])
    reasoning_parts.append(tail["reasoning"])
    return "".join(content_parts), "".join(reasoning_parts)


def test_plain_content_keeps_streaming_to_answer():
    answer, reasoning = collect_chunks(["你好", "，这是", "普通回答"])
    assert answer == "你好，这是普通回答"
    assert reasoning == ""


def test_think_content_is_split_out_of_answer_stream():
    answer, reasoning = collect_chunks(["<think>先分析", "一下</think>", "最终回答"])
    assert answer == "最终回答"
    assert reasoning == "先分析一下"


def test_partial_think_tags_across_chunks_are_supported():
    answer, reasoning = collect_chunks(["<thi", "nk>推理", "</th", "ink>结果"])
    assert answer == "结果"
    assert reasoning == "推理"


def test_unclosed_think_is_flushed_to_reasoning():
    answer, reasoning = collect_chunks(["开头", "<think>还在思考", "，没有闭合"])
    assert answer == "开头"
    assert reasoning == "还在思考，没有闭合"
