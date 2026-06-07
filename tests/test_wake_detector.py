from wake_word_detector.wake_detector import WakePhraseDetector


def test_detects_default_phrase_case_insensitive() -> None:
    detector = WakePhraseDetector(("computer",))

    event = detector.detect("Computer, what time is it?")

    assert event is not None
    assert event.phrase == "computer"


def test_word_boundary_prevents_partial_match() -> None:
    detector = WakePhraseDetector(("computer",))

    assert detector.detect("supercomputer status") is None


def test_detects_multi_word_phrase() -> None:
    detector = WakePhraseDetector(("ok assistant",))

    event = detector.detect("please OK   assistant start")

    assert event is not None
    assert event.phrase == "ok assistant"


def test_supports_multiple_phrases() -> None:
    detector = WakePhraseDetector(("computer", "ok assistant"))

    event = detector.detect("ok assistant")

    assert event is not None
    assert event.phrase == "ok assistant"


def test_fuzzy_detection_can_match_near_miss() -> None:
    detector = WakePhraseDetector(("computer",), fuzzy_threshold=1)

    event = detector.detect("commuter")

    assert event is not None
    assert event.fuzzy_distance == 1


def test_fuzzy_detection_respects_threshold() -> None:
    detector = WakePhraseDetector(("computer",), fuzzy_threshold=1)

    assert detector.detect("assistant") is None

