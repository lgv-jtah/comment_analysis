from context_builder import (
    ParentContext,
    build_comment_context,
    build_key_message_refs,
)


def test_build_key_message_refs_assigns_stable_positional_ids():
    refs = build_key_message_refs(
        "83",
        [
            {"key_message": "km one", "key_message_type": "Wunsch", "related_sentences": ["s1"]},
            {"key_message": "km two"},
        ],
    )

    assert [r.key_message_id for r in refs] == ["83:km1", "83:km2"]
    assert refs[0].text == "km one"
    assert refs[0].key_message_type == "Wunsch"
    assert refs[0].related_sentences == ["s1"]
    assert refs[1].key_message_type == ""
    assert refs[1].related_sentences == []


def test_build_key_message_refs_empty_list():
    assert build_key_message_refs("1", []) == []


def test_build_comment_context_full_context_available():
    context = build_comment_context(
        run_id="run-1",
        contribution_id="83",
        comment_text="Super Idee!",
        key_messages=[
            {"key_message": "km1", "related_sentences": ["a sentence"]},
        ],
        contribution_text="Der ganze Beitrag.",
        project_id="proj-a",
        comment_id="c1",
    )

    assert context.contribution_text_available is True
    # missing_parent_context is expected here too: no parent context source
    # exists yet (see docs/stance_policy_v2.md), so it's always flagged
    # unless the caller explicitly supplies one.
    assert context.context_flags == ["missing_parent_context"]
    assert context.parent_context.available is False
    assert len(context.key_messages) == 1
    assert context.key_messages[0].key_message_id == "83:km1"


def test_build_comment_context_flags_missing_contribution_text():
    context = build_comment_context(
        run_id="run-1",
        contribution_id="83",
        comment_text="text",
        key_messages=[{"key_message": "km1", "related_sentences": ["s"]}],
    )

    assert context.contribution_text_available is False
    assert "missing_contribution_text" in context.context_flags


def test_build_comment_context_flags_missing_related_sentences():
    context = build_comment_context(
        run_id="run-1",
        contribution_id="83",
        comment_text="text",
        key_messages=[{"key_message": "km1"}, {"key_message": "km2"}],
        contribution_text="full text",
    )

    assert "missing_related_sentences" in context.context_flags


def test_build_comment_context_flags_no_key_messages():
    context = build_comment_context(
        run_id="run-1",
        contribution_id="83",
        comment_text="text",
        key_messages=[],
        contribution_text="full text",
    )

    assert "no_key_messages" in context.context_flags


def test_build_comment_context_default_parent_context_is_unavailable_and_flagged():
    context = build_comment_context(
        run_id="run-1",
        contribution_id="83",
        comment_text="text",
        key_messages=[{"key_message": "km1", "related_sentences": ["s"]}],
        contribution_text="full text",
    )

    assert context.parent_context == ParentContext(available=False)
    assert "missing_parent_context" in context.context_flags


def test_build_comment_context_explicit_parent_context_not_flagged():
    parent = ParentContext(available=True, parent_comment_text="prior comment", parent_kind="comment")

    context = build_comment_context(
        run_id="run-1",
        contribution_id="83",
        comment_text="text",
        key_messages=[{"key_message": "km1", "related_sentences": ["s"]}],
        contribution_text="full text",
        parent_context=parent,
    )

    assert context.parent_context.available is True
    assert "missing_parent_context" not in context.context_flags
