import pytest

from duk.tts import _normalize_numbers, _normalize_traditional, _nfkc_preserve_ellipsis


def norm(text: str) -> str:
    """Run the same pre-passes _prepare_tts_pipeline applies before numbers."""
    return _normalize_numbers(_normalize_traditional(_nfkc_preserve_ellipsis(text)))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Full-width "－" folds to ASCII "-" under NFKC; both endpoints are years.
        ("1368－1643年", "一三六八到一六四三年"),
        ("1368-1643年", "一三六八到一六四三年"),
        # 至/到 are already words and survive; only the reading changes.
        ("1937至1939年", "一九三七至一九三九年"),
        ("1828到1833年", "一八二八到一八三三年"),
        # A two-digit tail abbreviates the left year rather than naming a duration.
        ("1977－78年度", "一九七七到七八年度"),
        ("1978－81年", "一九七八到八一年"),
        # Ancient years keep the digit-wise reading the 年 rule already gave them.
        ("618－907年", "六一八到九零七年"),
    ],
)
def test_year_ranges(raw: str, expected: str) -> None:
    assert norm(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("（1898-1969）", "(一八九八到一九六九)"),
        ("（1937-1939）", "(一九三七到一九三九)"),
    ],
)
def test_bare_year_ranges_in_citations(raw: str, expected: str) -> None:
    assert norm(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("公元前221－公元前206年", "公元前二二一到公元前二零六年"),
        ("公元前202年－公元後220年", "公元前二零二年到公元后二二零年"),
        # The right endpoint may drop the era marker.
        ("公元前373－288年", "公元前三七三到二八八年"),
    ],
)
def test_era_ranges(raw: str, expected: str) -> None:
    assert norm(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("8－10月", "八到十月"),
        ("28－29日", "二十八到二十九日"),
        ("220－370億美元", "二百二十到三百七十亿美元"),
        ("3~5天", "三到五天"),
        # Repeated unit between the endpoints.
        ("1,050萬－1,200萬噸", "一千零五十万到一千二百万吨"),
    ],
)
def test_quantity_ranges(raw: str, expected: str) -> None:
    assert norm(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # One 百分之 in front, not one per endpoint.
        ("10－15％", "百分之十到十五"),
        ("60－90％", "百分之六十到九十"),
        ("50.9％", "百分之五十点九"),
    ],
)
def test_percent_ranges(raw: str, expected: str) -> None:
    assert norm(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,000兩", "一千两"),
        ("代表1,936人", "代表一千九百三十六人"),
        ("231,910件", "二十三万一千九百一十件"),
        ("17,000億元", "一万七千亿元"),
    ],
)
def test_thousands_separators(raw: str, expected: str) -> None:
    assert norm(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Short digit runs before 年 are durations, not calendar years.
        ("凡39年", "凡三十九年"),
        ("租期為99年", "租期为九十九年"),
        ("25年的權利", "二十五年的权利"),
        # Four-digit years stay digit-wise.
        ("1984年", "一九八四年"),
    ],
)
def test_durations_versus_years(raw: str, expected: str) -> None:
    assert norm(raw) == expected


def test_archive_code_is_not_a_range() -> None:
    """Three-part hyphen chains are citation codes; no 到 may be inserted."""
    assert "到" not in norm("123-25-2")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2024-05-01", "二零二四年五月一日"),
        ("14:30", "十四点三十分"),
        ("3.14", "三点一四"),
        ("1/2", "二分之一"),
    ],
)
def test_existing_shapes_still_win_over_ranges(raw: str, expected: str) -> None:
    assert norm(raw) == expected
