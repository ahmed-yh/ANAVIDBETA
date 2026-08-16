from pipeline import parse_person_id


def test_parse_plain_id():
    assert parse_person_id("45") == 45


def test_parse_visit_suffixed_id():
    assert parse_person_id("45_visit1") == 45
    assert parse_person_id("45_visit2") == 45


def test_parse_int_key():
    assert parse_person_id(45) == 45


def test_parse_invalid_key_returns_none():
    assert parse_person_id("not_a_number") is None
