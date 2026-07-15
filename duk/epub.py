from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from posixpath import dirname as posix_dirname
from posixpath import join as posix_join
from posixpath import normpath as posix_normpath
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import unquote

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, ITEM_IMAGE, ITEM_STYLE, epub

from .text import SECTION_BREAK, normalize_section_breaks


@dataclass(frozen=True)
class TocEntry:
    title: str
    href: str


@dataclass(frozen=True)
class Chapter:
    title: str
    href: str
    source: str
    text: str
    headings: List[str] = field(default_factory=list)
    heading_categories: dict[str, str] = field(default_factory=dict)


def read_epub(path: Path) -> epub.EpubBook:
    return epub.read_epub(str(path))


def _first_dc_meta(book: epub.EpubBook, name: str) -> str:
    items = book.get_metadata("DC", name)
    if not items:
        return ""
    value, _attrs = items[0]
    return value or ""


def _all_dc_meta(book: epub.EpubBook, name: str) -> List[str]:
    items = book.get_metadata("DC", name)
    values: List[str] = []
    for value, _attrs in items:
        if value:
            values.append(value)
    return values


def _item_name(item: object) -> str:
    get_name = getattr(item, "get_name", None)
    if callable(get_name):
        return get_name() or ""
    return getattr(item, "file_name", "") or ""


def _item_title(item: object) -> str:
    title = getattr(item, "title", "")
    if title:
        return title
    get_title = getattr(item, "get_title", None)
    if callable(get_title):
        value = get_title()
        if value:
            return value
    return ""


def _item_id(item: object) -> str:
    get_id = getattr(item, "get_id", None)
    if callable(get_id):
        value = get_id()
        if value:
            return value
    return getattr(item, "id", "") or ""


def _find_cover_item(book: epub.EpubBook) -> object | None:
    cover_meta = book.get_metadata("OPF", "cover")
    if cover_meta:
        _value, attrs = cover_meta[0]
        cover_id = attrs.get("content") if attrs else None
        if cover_id:
            item = book.get_item_with_id(cover_id)
            if item:
                return item

    opf_meta = book.get_metadata("OPF", "meta")
    for _value, attrs in opf_meta:
        if not attrs:
            continue
        if str(attrs.get("name") or "").lower() != "cover":
            continue
        cover_id = attrs.get("content")
        if cover_id:
            item = book.get_item_with_id(cover_id)
            if item:
                return item

    for item in book.get_items():
        props = getattr(item, "properties", []) or []
        if isinstance(props, str):
            props = [props]
        if any("cover-image" in prop for prop in props):
            return item

    for item in book.get_items_of_type(ITEM_IMAGE):
        name = _item_name(item).lower()
        item_id = _item_id(item).lower()
        if "cover" in name or "cover" in item_id:
            return item

    best_item = None
    best_size = 0
    for item in book.get_items_of_type(ITEM_IMAGE):
        try:
            data = item.get_content()
        except Exception:
            continue
        if not data:
            continue
        size = len(data)
        if size > best_size:
            best_size = size
            best_item = item

    if best_item:
        return best_item

    return None


def extract_metadata(book: epub.EpubBook) -> dict:
    title = _first_dc_meta(book, "title")
    authors = _all_dc_meta(book, "creator")
    language = _first_dc_meta(book, "language")
    dates = _all_dc_meta(book, "date")
    year = ""
    for value in dates:
        match = re.search(r"(19|20)\d{2}", value)
        if match:
            year = match.group(0)
            break

    cover_info = None
    cover_item = _find_cover_item(book)
    if cover_item:
        cover_info = {
            "id": _item_id(cover_item),
            "href": _item_name(cover_item),
            "media_type": getattr(cover_item, "media_type", "") or "",
        }

    return {
        "title": title,
        "authors": authors,
        "language": language,
        "dates": dates,
        "year": year,
        "cover": cover_info,
    }


def extract_cover_image(book: epub.EpubBook) -> dict | None:
    cover_item = _find_cover_item(book)
    if not cover_item:
        return None
    data = cover_item.get_content()
    if not data:
        return None
    return {
        "data": data,
        "id": _item_id(cover_item),
        "href": _item_name(cover_item),
        "media_type": getattr(cover_item, "media_type", "") or "",
    }


def normalize_href(href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    href = href.split("#", 1)[0]
    # Some EPUBs percent-encode filenames in TOC entries.
    return unquote(href)


def flatten_toc(toc: Iterable) -> List[TocEntry]:
    entries: List[TocEntry] = []

    def walk(nodes: Iterable) -> None:
        for node in nodes:
            if isinstance(node, epub.Link):
                if node.href:
                    entries.append(TocEntry(title=node.title or "", href=node.href))
            elif isinstance(node, epub.Section):
                if node.href:
                    entries.append(TocEntry(title=node.title or "", href=node.href))
                subitems = getattr(node, "subitems", None)
                if subitems:
                    walk(subitems)
            elif isinstance(node, (list, tuple)):
                walk(node)

    walk(toc)
    return entries


def build_toc_entries(book: epub.EpubBook) -> List[TocEntry]:
    toc = book.toc or []
    entries = flatten_toc(toc)
    return [e for e in entries if e.href]


def build_spine_entries(book: epub.EpubBook) -> List[TocEntry]:
    entries: List[TocEntry] = []
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if not item or item.get_type() != ITEM_DOCUMENT:
            continue
        title = _item_title(item) or _item_name(item) or ""
        entries.append(TocEntry(title=title, href=_item_name(item)))
    return entries


def _build_spine_items(book: epub.EpubBook) -> List[tuple[str, object]]:
    items: List[tuple[str, object]] = []
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if not item or item.get_type() != ITEM_DOCUMENT:
            continue
        name = _item_name(item)
        if not name:
            continue
        items.append((normalize_href(name), item))
    return items


def _split_series_key(href: str) -> tuple[str, str] | None:
    match = re.match(r"^(?P<prefix>.+?)(?:_split_|-split-)\d+(?P<ext>\.[^./]+)$", href)
    if not match:
        return None
    return (match.group("prefix"), match.group("ext"))


def _join_item_text(
    items: Iterable[object], footnote_index: dict[str, set[str]] | None = None
) -> str:
    parts: List[str] = []
    for item in items:
        text = html_to_text(
            item.get_content(),
            footnote_index=footnote_index,
            source_href=_item_name(item),
        )
        if text:
            parts.append(text)
    if not parts:
        return ""
    return normalize_text("\n\n".join(parts))


_NOTE_MARKER_RE = re.compile(r"^\d{1,4}[a-z]?[.)]?$", re.IGNORECASE)


def _normalize_id(value: str) -> str:
    return (value or "").strip().lower()


def _href_fragment(href: str) -> str:
    if not href:
        return ""
    parts = href.split("#", 1)
    if len(parts) < 2:
        return ""
    return unquote(parts[1])


def _looks_like_note_marker(text: str) -> bool:
    if not text:
        return False
    return bool(_NOTE_MARKER_RE.match(text.strip()))


_NAV_CONTAINER_EPUB_TYPES = {"toc", "page-list", "landmarks", "loi", "lot"}


def _is_navigation_anchor(anchor: object) -> bool:
    """TOC/page-list links routinely carry bare numbers as labels (numbered
    chapters, page anchors), so navigation anchors must never seed the
    footnote-marker heuristics."""
    for parent in getattr(anchor, "parents", None) or ():
        name = str(getattr(parent, "name", "") or "").lower()
        if name == "nav":
            return True
        get = getattr(parent, "get", None)
        if not callable(get):
            continue
        epub_type = str(get("epub:type") or "").strip().lower()
        if epub_type in _NAV_CONTAINER_EPUB_TYPES:
            return True
    return False


def _collect_footnote_index(
    book: epub.EpubBook,
) -> dict[str, set[str]]:
    note_ids: set[str] = set()
    backref_ids: set[str] = set()

    for item in book.get_items_of_type(ITEM_DOCUMENT):
        content = item.get_content()
        if not content:
            continue
        head = content.lstrip()[:512].lower()
        parser = (
            "lxml-xml"
            if (head.startswith(b"<?xml") or b"xmlns=" in head)
            else "lxml"
        )
        soup = BeautifulSoup(content, parser)
        for tag in soup.find_all(attrs={"epub:type": "footnote"}):
            note_id = _normalize_id(str(tag.get("id") or ""))
            if note_id:
                note_ids.add(note_id)
        for tag in soup.find_all(attrs={"role": "doc-footnote"}):
            note_id = _normalize_id(str(tag.get("id") or ""))
            if note_id:
                note_ids.add(note_id)
        for tag in soup.find_all(["p", "section", "div", "aside", "ol", "ul", "li", "td"]):
            attrs = getattr(tag, "attrs", None)
            if attrs is None:
                continue
            classes = attrs.get("class", [])
            if isinstance(classes, str):
                classes = [classes]
            class_text = " ".join(classes).lower()
            id_text = str(attrs.get("id") or "")
            if "footnote" in class_text or "endnote" in class_text:
                note_id = _normalize_id(id_text)
                if note_id:
                    note_ids.add(note_id)
            if id_text.lower().startswith(("fn", "footnote", "endnote")):
                note_id = _normalize_id(id_text)
                if note_id:
                    note_ids.add(note_id)

        for anchor in soup.find_all("a"):
            if _is_navigation_anchor(anchor):
                continue
            text = anchor.get_text(strip=True)
            if not _looks_like_note_marker(text):
                continue
            fragment = _normalize_id(_href_fragment(str(anchor.get("href") or "")))
            if fragment:
                backref_ids.add(fragment)
            parent = anchor.parent
            if parent and getattr(parent, "attrs", None):
                parent_id = _normalize_id(str(parent.get("id") or ""))
                if parent_id:
                    parent_text = parent.get_text(strip=True)
                    if parent_text == text:
                        note_ids.add(parent_id)

    return {"note_ids": note_ids, "backref_ids": backref_ids}


def _parse_html_soup(html: bytes | str) -> BeautifulSoup:
    if isinstance(html, bytes):
        head = html.lstrip()[:512].lower()
        parser = (
            "lxml-xml"
            if (head.startswith(b"<?xml") or b"xmlns=" in head)
            else "lxml"
        )
    else:
        head = str(html).lstrip()[:512].lower()
        parser = "lxml-xml" if (head.startswith("<?xml") or "xmlns=" in head) else "lxml"
    return BeautifulSoup(html, parser)


def _render_image_alt_text(soup: BeautifulSoup) -> None:
    for img in soup.find_all("img"):
        alt = (img.get("alt") or "").strip()
        if not alt:
            continue
        img.replace_with(alt)


_BACKREF_KEEP_MIN_CHARS = 120
_BACKREF_KEEP_MIN_PAGE_RATIO = 0.8


def _holds_most_of_page_text(tag: object, page_text_len: int) -> bool:
    """Backref fragments sometimes resolve to a page's root container (e.g. a
    chapter <section> targeted by page anchors) rather than a note marker.
    A real footnote is subordinate content, so an element holding nearly all
    of a substantial page must be page content and may not be removed."""
    if page_text_len <= 0:
        return False
    get_text = getattr(tag, "get_text", None)
    if not callable(get_text):
        return False
    tag_len = len(_compact_heading_text(get_text()))
    if tag_len < _BACKREF_KEEP_MIN_CHARS:
        return False
    return tag_len >= page_text_len * _BACKREF_KEEP_MIN_PAGE_RATIO


def _soup_to_text(
    soup: BeautifulSoup,
    footnote_index: dict[str, set[str]] | None = None,
    source_href: str = "",
) -> str:
    _ = source_href
    _render_image_alt_text(soup)
    # Ruby in Chinese EPUBs annotates pronunciation (zhuyin/pinyin); keep the
    # base text and drop the annotation glyphs.
    for tag in soup.find_all(["rt", "rp"]):
        tag.decompose()
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()
    for tag in soup.find_all("sup"):
        tag.decompose()
    for tag in soup.find_all(attrs={"epub:type": "noteref"}):
        tag.decompose()
    for tag in soup.find_all(attrs={"epub:type": "footnote"}):
        tag.decompose()
    for tag in soup.find_all(attrs={"role": "doc-noteref"}):
        tag.decompose()
    for tag in soup.find_all(attrs={"role": "doc-footnote"}):
        tag.decompose()
    if footnote_index:
        note_ids = footnote_index.get("note_ids", set())
        backref_ids = footnote_index.get("backref_ids", set())
        if backref_ids:
            page_root = soup.body if soup.body else soup
            page_text_len = len(_compact_heading_text(page_root.get_text()))
            for tag in soup.find_all(attrs={"id": True}):
                attrs = getattr(tag, "attrs", None)
                if not attrs:
                    continue
                tag_id = _normalize_id(str(attrs.get("id") or ""))
                if not tag_id or tag_id not in backref_ids:
                    continue
                if _holds_most_of_page_text(tag, page_text_len):
                    continue
                tag.decompose()
        if note_ids or backref_ids:
            for anchor in soup.find_all("a"):
                text = anchor.get_text(strip=True)
                if not _looks_like_note_marker(text):
                    continue
                fragment = _normalize_id(
                    _href_fragment(str(anchor.get("href") or ""))
                )
                if fragment and (fragment in note_ids or fragment in backref_ids):
                    anchor.decompose()
    for tag in soup.find_all(["p", "section", "div", "aside", "ol", "ul", "li"]):
        attrs = getattr(tag, "attrs", None)
        if attrs is None:
            continue
        classes = attrs.get("class", [])
        if isinstance(classes, str):
            classes = [classes]
        class_text = " ".join(classes).lower()
        id_text = str(attrs.get("id") or "").lower()
        if (
            "footnote" in class_text
            or "endnote" in class_text
            or "copyright" in class_text
            or "credit" in class_text
        ):
            tag.decompose()
            continue
        if id_text.startswith(("fn", "footnote", "endnote")):
            tag.decompose()
    root = soup.body if soup.body else soup

    block_tags = {
        "p",
        "div",
        "li",
        "blockquote",
        "pre",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    def _class_key(tag: object) -> str:
        classes = getattr(tag, "get", lambda *_args, **_kwargs: [])("class", [])
        if isinstance(classes, str):
            classes = [classes]
        cleaned = [str(c).strip() for c in classes if str(c).strip()]
        return " ".join(cleaned)

    def _top_level_blocks(name: str) -> List[object]:
        out: List[object] = []
        for elem in root.find_all(name):
            if any(getattr(parent, "name", None) in block_tags for parent in elem.parents):
                continue
            out.append(elem)
        return out

    p_class_counts: dict[str, int] = {}
    for para in _top_level_blocks("p"):
        key = _class_key(para)
        if not key:
            continue
        text = para.get_text(separator="", strip=True)
        if not text:
            continue
        p_class_counts[key] = p_class_counts.get(key, 0) + 1

    normal_classes: set[str] = set()
    if p_class_counts:
        primary_count = max(p_class_counts.values())
        if primary_count >= 8:
            threshold = max(3, int(primary_count * 0.2))
            normal_classes = {
                key for key, count in p_class_counts.items() if count >= threshold
            }
        else:
            normal_classes = set(p_class_counts.keys())

    ornament_re = re.compile(
        r"^(?:[＊*・…‥…\-—=]{3,}|[◆◇■□●○]{3,}|"
        r"(?:\*|＊)(?:\s*(?:\*|＊)){2,})$"
    )

    blocks: List[str] = []
    for elem in root.find_all(block_tags):
        if any(getattr(parent, "name", None) in block_tags for parent in elem.parents):
            continue
        if getattr(elem, "name", None) == "hr":
            if blocks:
                blocks.append(SECTION_BREAK)
            continue

        for br in elem.find_all("br"):
            br.replace_with("\n")
        text = elem.get_text(separator="", strip=False)
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = text.strip()
        if not text:
            continue

        if ornament_re.fullmatch(text):
            if blocks:
                blocks.append(SECTION_BREAK)
            continue

        if getattr(elem, "name", None) == "p":
            key = _class_key(elem)
            if key and key not in normal_classes and blocks:
                blocks.append(SECTION_BREAK)

        blocks.append(text)

    if blocks:
        return normalize_text("\n\n".join(blocks))

    text = root.get_text(separator="", strip=False)
    return normalize_text(text)


def html_to_text(
    html: bytes | str,
    footnote_index: dict[str, set[str]] | None = None,
    source_href: str = "",
) -> str:
    soup = _parse_html_soup(html)
    return _soup_to_text(soup, footnote_index=footnote_index, source_href=source_href)


def normalize_text(text: str) -> str:
    text = (
        text.replace("ʼ", "'")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("«", '"')
        .replace("»", '"')
    )
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    if "\n\n\n" in text:
        text = re.sub(r"\n{3,}", f"\n\n{SECTION_BREAK}\n\n", text)
    text = normalize_section_breaks(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\.[ \t]*\.[ \t]*\.", "...", text)
    return text.strip()


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:60] or "chapter"


_TITLE_MAX_LEN = 40
_TITLE_ONLY_CHAPTER_MAX_LEN = 80
_TITLE_ONLY_MERGE_MIN_NEXT_TEXT_LEN = 400
_NON_CONTENT_TOC_TITLES = {
    "封面",
    "封底",
    "目录",
    "目錄",
    "版权",
    "版權",
    "版权页",
    "版權頁",
    "版权信息",
    "版權資訊",
    "版權信息",
    "扉页",
    "扉頁",
    "contents",
    "toc",
    "cover",
    "copyright",
}
_NON_CONTENT_TOC_TITLES_CASEFOLD = {value.casefold() for value in _NON_CONTENT_TOC_TITLES}
_INLINE_TOC_MIN_LINKS = 3
_TOC_SPINE_COVERAGE_MIN_RATIO = 0.9


def _title_from_text(text: str, max_len: int = _TITLE_MAX_LEN) -> str:
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if len(cleaned) > max_len:
            return cleaned[:max_len].rstrip() + "..."
        return cleaned
    return ""


def _extract_heading_title(
    html: bytes | str,
    heading_class_markers: Optional[set[str]] = None,
    heading_id_markers: Optional[set[str]] = None,
) -> str:
    headings = _extract_html_headings(
        html,
        heading_class_markers=heading_class_markers,
        heading_id_markers=heading_id_markers,
    )
    if headings:
        return headings[0]

    soup = _parse_html_soup(html)
    title_node = soup.find("title")
    if title_node:
        text = title_node.get_text(separator=" ", strip=True)
        if text:
            return text
    return ""


_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_HEADING_CLASS_HINTS = (
    "title",
    "subtitle",
    "chapter",
    "section",
    "headline",
    "heading",
)
# Chinese structural labels: 第N章/回/节/話..., bare numerals, roman numerals.
_CN_UNIT_CHARS = "章節节話话部卷巻回篇編编集幕讲講"
_CN_NUMERAL_CHARS = "0-9０-９一二三四五六七八九十百千〇零两兩"
_SHORT_HEADING_LABEL_RE = re.compile(
    rf"^(?:"
    rf"第?[{_CN_NUMERAL_CHARS}]+(?:[{_CN_UNIT_CHARS}])?"
    rf"|[{_CN_NUMERAL_CHARS}]+(?:[.)．、])?"
    rf"|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+(?:[.)．、])?"
    rf"|[IVXLCDM]+(?:[.)．、])?"
    rf")$"
)
_STRUCTURAL_TITLE_PREFIX_RE = re.compile(
    rf"^(?:"
    rf"第?[{_CN_NUMERAL_CHARS}]+(?:[{_CN_UNIT_CHARS}])"
    rf"|[{_CN_NUMERAL_CHARS}]+(?:[{_CN_UNIT_CHARS}]|[.)．、])"
    rf"|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVXLCDM]+(?:[{_CN_UNIT_CHARS}]|[.)．、])"
    rf")"
)
_STRUCTURAL_HEADING_TERMS_CASEFOLD = {
    "prologue",
    "epilogue",
    "postscript",
    "interlude",
    "preface",
    "afterword",
    "introduction",
    "序",
    "序章",
    "序言",
    "序曲",
    "自序",
    "代序",
    "前言",
    "引言",
    "引子",
    "楔子",
    "导言",
    "導言",
    "导读",
    "導讀",
    "尾声",
    "尾聲",
    "终章",
    "終章",
    "结语",
    "結語",
    "结束语",
    "結束語",
    "后记",
    "後記",
    "跋",
    "番外",
    "附录",
    "附錄",
}
_CSS_SELECTOR_RULE_RE = re.compile(r"([^{}]+)\{")
_CSS_CLASS_OR_ID_RE = re.compile(r"([.#])([A-Za-z_][A-Za-z0-9_-]*)")
_BLOCK_TEXT_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "li",
    "dt",
    "dd",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "td",
    "th",
    "caption",
}
_INLINE_TEXT_TAGS = {"span", "a", "em", "strong", "b", "i", "u", "small", "sup", "sub"}
_HEADING_CATEGORY_SECTION = "section"
_HEADING_CATEGORY_TITLE = "title"


def _looks_like_heading_marker(
    value: str, explicit_markers: Optional[set[str]] = None
) -> bool:
    cleaned = str(value or "").strip().lower()
    if not cleaned:
        return False
    if explicit_markers and cleaned in explicit_markers:
        return True
    if any(token in cleaned for token in _HEADING_CLASS_HINTS):
        return True
    return bool(re.search(r"(^|[-_])(ttl|hd)([-_0-9]|$)", cleaned))


def _normalize_heading_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_heading_key(value: str) -> str:
    return _normalize_heading_text(value).casefold()


def _heading_category_priority(value: str) -> int:
    if value == _HEADING_CATEGORY_TITLE:
        return 2
    return 1


def _merge_heading_category(current: str, candidate: str) -> str:
    if _heading_category_priority(candidate) > _heading_category_priority(current):
        return candidate
    return current


def _heading_category_from_marker_name(marker: str) -> str:
    cleaned = str(marker or "").strip().lower()
    if not cleaned:
        return _HEADING_CATEGORY_SECTION
    if "subtitle" in cleaned:
        return _HEADING_CATEGORY_SECTION
    if "title" in cleaned:
        return _HEADING_CATEGORY_TITLE
    return _HEADING_CATEGORY_SECTION


def _heading_category_from_tag_name(tag: str) -> str:
    cleaned = str(tag or "").strip().lower()
    if cleaned == "h1":
        return _HEADING_CATEGORY_TITLE
    return _HEADING_CATEGORY_SECTION


def _normalized_node_text(node: object) -> str:
    text = getattr(node, "get_text", lambda *_args, **_kwargs: "")(
        separator=" ", strip=True
    )
    return _normalize_heading_text(text)


def _compact_heading_text(value: str) -> str:
    return re.sub(r"[\s　]+", "", value)


def _looks_like_short_structural_heading(value: str) -> bool:
    cleaned = _normalize_heading_text(value)
    if not cleaned or len(cleaned) > 40:
        return False
    compact = _compact_heading_text(cleaned)
    if len(compact) > 20:
        return False
    return bool(_SHORT_HEADING_LABEL_RE.fullmatch(compact))


def _looks_like_structural_heading_term(value: str) -> bool:
    cleaned = _normalize_heading_text(value)
    if not cleaned:
        return False
    if cleaned.casefold() in _STRUCTURAL_HEADING_TERMS_CASEFOLD:
        return True
    compact = _compact_heading_text(cleaned)
    if compact.casefold() in _STRUCTURAL_HEADING_TERMS_CASEFOLD:
        return True
    return False


def _looks_like_structural_title(value: str) -> bool:
    cleaned = _normalize_heading_text(value)
    if not cleaned:
        return False
    if _looks_like_short_structural_heading(cleaned):
        return True
    if _looks_like_structural_heading_term(cleaned):
        return True
    compact = _compact_heading_text(cleaned)
    if not compact or len(compact) > 80:
        return False
    return bool(_STRUCTURAL_TITLE_PREFIX_RE.match(compact))


def _looks_like_body_text_block(value: str) -> bool:
    cleaned = _normalize_heading_text(value)
    if not cleaned:
        return False
    compact = _compact_heading_text(cleaned)
    if len(compact) >= 20:
        return True
    if len(compact) >= 8 and any(mark in cleaned for mark in ("。", "．", "!", "?", "！", "？")):
        return True
    return False


def _is_standalone_heading_node(node: object, text: str) -> bool:
    name = str(getattr(node, "name", "") or "").lower()
    if name in _BLOCK_TEXT_TAGS:
        return True
    if name not in _INLINE_TEXT_TAGS:
        return False
    parent = getattr(node, "parent", None)
    if not parent:
        return False
    parent_name = str(getattr(parent, "name", "") or "").lower()
    if parent_name not in _BLOCK_TEXT_TAGS:
        return False
    return _normalized_node_text(parent) == text


def _decode_css_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "gb18030", "big5"):
        try:
            return data.decode(encoding)
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")


def _collect_css_heading_markers(book: epub.EpubBook) -> tuple[set[str], set[str]]:
    class_markers: set[str] = set()
    id_markers: set[str] = set()
    for item in book.get_items():
        name = _item_name(item).lower()
        if item.get_type() != ITEM_STYLE and not name.endswith(".css"):
            continue
        content = item.get_content()
        if not content:
            continue
        css_text = _decode_css_text(content)
        for selector in _CSS_SELECTOR_RULE_RE.findall(css_text):
            for marker_type, marker_name in _CSS_CLASS_OR_ID_RE.findall(selector):
                cleaned = marker_name.strip().lower()
                if not _looks_like_heading_marker(cleaned):
                    continue
                if marker_type == ".":
                    class_markers.add(cleaned)
                else:
                    id_markers.add(cleaned)
    return class_markers, id_markers


def _collect_structural_heading_classes(book: epub.EpubBook) -> set[str]:
    class_totals: dict[str, int] = {}
    class_heading_like: dict[str, int] = {}
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        content = item.get_content()
        if not content:
            continue
        soup = _parse_html_soup(content)
        root = soup.body if soup.body else soup
        for node in root.find_all(True):
            classes = getattr(node, "get", lambda *_args, **_kwargs: [])("class", [])
            if isinstance(classes, str):
                classes = [classes]
            cleaned_classes = [str(cls).strip().lower() for cls in classes if str(cls).strip()]
            if not cleaned_classes:
                continue
            text = _normalized_node_text(node)
            heading_like = (
                _looks_like_short_structural_heading(text)
                and _is_standalone_heading_node(node, text)
            )
            for cls in cleaned_classes:
                class_totals[cls] = class_totals.get(cls, 0) + 1
                if heading_like:
                    class_heading_like[cls] = class_heading_like.get(cls, 0) + 1

    inferred: set[str] = set()
    for cls, total in class_totals.items():
        hits = class_heading_like.get(cls, 0)
        if hits < 4:
            continue
        if (hits / total) >= 0.8:
            inferred.add(cls)
    return inferred


def _collect_book_heading_markers(book: epub.EpubBook) -> tuple[set[str], set[str]]:
    class_markers, id_markers = _collect_css_heading_markers(book)
    class_markers.update(_collect_structural_heading_classes(book))
    return class_markers, id_markers


def _heading_categories_from_entries(
    entries: List[tuple[str, str]],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for text, category in entries:
        key = _normalize_heading_key(text)
        if not key:
            continue
        previous = out.get(key)
        if previous is None:
            out[key] = category
            continue
        out[key] = _merge_heading_category(previous, category)
    return out


def _extract_html_heading_entries(
    html: bytes | str,
    heading_class_markers: Optional[set[str]] = None,
    heading_id_markers: Optional[set[str]] = None,
) -> List[tuple[str, str]]:
    class_markers = {marker.strip().lower() for marker in (heading_class_markers or set())}
    id_markers = {marker.strip().lower() for marker in (heading_id_markers or set())}

    soup = _parse_html_soup(html)
    _render_image_alt_text(soup)
    for tag in soup.find_all(["rt", "rp", "script", "style"]):
        tag.decompose()
    root = soup.body if soup.body else soup
    out: List[tuple[str, str]] = []
    seen: dict[str, int] = {}

    def add_node_text(node: object, category: str) -> None:
        cleaned = _normalized_node_text(node)
        if not cleaned or len(cleaned) > 120:
            return
        key = cleaned.casefold()
        idx = seen.get(key)
        if idx is not None:
            prev_text, prev_category = out[idx]
            merged = _merge_heading_category(prev_category, category)
            if merged != prev_category:
                out[idx] = (prev_text, merged)
            return
        seen[key] = len(out)
        out.append((cleaned, category))

    for tag in _HEADING_TAGS:
        category = _heading_category_from_tag_name(tag)
        for node in root.find_all(tag):
            add_node_text(node, category)

    for node in root.find_all(True):
        name = str(getattr(node, "name", "") or "").lower()
        if name in _HEADING_TAGS:
            continue
        classes = getattr(node, "get", lambda *_args, **_kwargs: [])("class", [])
        if isinstance(classes, str):
            classes = [classes]
        id_value = str(
            getattr(node, "get", lambda *_args, **_kwargs: "")("id", "") or ""
        )
        marker_categories: List[str] = []
        for cls in classes:
            cls_text = str(cls or "").strip()
            if not _looks_like_heading_marker(cls_text, class_markers):
                continue
            marker_categories.append(_heading_category_from_marker_name(cls_text))
        if _looks_like_heading_marker(id_value, id_markers):
            marker_categories.append(_heading_category_from_marker_name(id_value))
        if not marker_categories:
            continue
        category = _HEADING_CATEGORY_SECTION
        for marker_category in marker_categories:
            category = _merge_heading_category(category, marker_category)
        add_node_text(node, category)

    return out


def _extract_html_headings(
    html: bytes | str,
    heading_class_markers: Optional[set[str]] = None,
    heading_id_markers: Optional[set[str]] = None,
) -> List[str]:
    return [
        text
        for text, _category in _extract_html_heading_entries(
            html,
            heading_class_markers=heading_class_markers,
            heading_id_markers=heading_id_markers,
        )
    ]


def _extract_html_headings_and_categories(
    html: bytes | str,
    heading_class_markers: Optional[set[str]] = None,
    heading_id_markers: Optional[set[str]] = None,
) -> tuple[List[str], dict[str, str]]:
    entries = _extract_html_heading_entries(
        html,
        heading_class_markers=heading_class_markers,
        heading_id_markers=heading_id_markers,
    )
    headings = [text for text, _category in entries]
    categories = _heading_categories_from_entries(entries)
    return headings, categories


def _merge_heading_categories(
    base: dict[str, str], incoming: dict[str, str]
) -> dict[str, str]:
    merged = dict(base)
    for key, category in incoming.items():
        if key not in merged:
            merged[key] = category
            continue
        merged[key] = _merge_heading_category(merged[key], category)
    return merged


def _merge_headings_with_categories(
    headings: List[str],
    categories: dict[str, str],
    incoming_headings: List[str],
    incoming_categories: dict[str, str],
) -> tuple[List[str], dict[str, str]]:
    seen = {_normalize_heading_key(value) for value in headings}
    for heading in incoming_headings:
        key = _normalize_heading_key(heading)
        if not key or key in seen:
            continue
        seen.add(key)
        headings.append(heading)
    categories = _merge_heading_categories(categories, incoming_categories)
    return headings, categories


def _heading_categories_for_values(
    headings: List[str], categories: dict[str, str]
) -> dict[str, str]:
    out: dict[str, str] = {}
    for heading in headings:
        key = _normalize_heading_key(heading)
        if not key:
            continue
        category = categories.get(key, _HEADING_CATEGORY_SECTION)
        out[key] = category
    return out


def _apply_chapter_title_heading_category(
    chapter_title: str,
    headings: List[str],
    categories: dict[str, str],
) -> dict[str, str]:
    title_key = _normalize_heading_key(chapter_title)
    if not title_key:
        return categories
    if title_key not in {_normalize_heading_key(heading) for heading in headings}:
        return categories
    out = dict(categories)
    out[title_key] = _HEADING_CATEGORY_TITLE
    return out


def _looks_like_filename(title: str, href: str) -> bool:
    cleaned = str(title or "").strip()
    if not cleaned:
        return True
    base = Path(href or "")
    stem = base.stem
    name = base.name
    if cleaned == href or cleaned == name or cleaned == stem:
        return True
    lowered = cleaned.lower()
    if lowered.endswith(".xhtml") or lowered.endswith(".html"):
        return True
    if stem and lowered == stem.lower():
        return True
    if re.fullmatch(r"[a-z]\d+[a-z0-9]*", lowered):
        return True
    return False


def _is_non_content_toc_title(title: str) -> bool:
    cleaned = _normalize_heading_text(title).casefold()
    if not cleaned:
        return True
    return cleaned in _NON_CONTENT_TOC_TITLES_CASEFOLD


def _resolve_relative_href(base_href: str, target_href: str) -> str:
    target = normalize_href(target_href)
    if not target:
        return ""
    if "://" in target:
        return ""
    if target.startswith("/"):
        return target.lstrip("/")
    base = normalize_href(base_href)
    if not base:
        return target
    base_dir = posix_dirname(base)
    if not base_dir:
        return target
    return posix_normpath(posix_join(base_dir, target))


def _chapter_nonempty_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _chapter_first_nonempty_line(text: str) -> str:
    lines = _chapter_nonempty_lines(text)
    if not lines:
        return ""
    return lines[0]


def _is_title_only_chapter(chapter: Chapter) -> bool:
    lines = _chapter_nonempty_lines(chapter.text)
    if len(lines) != 1:
        return False
    only_line = lines[0]
    if len(only_line) > _TITLE_ONLY_CHAPTER_MAX_LEN:
        return False
    return _normalize_heading_text(chapter.title) == _normalize_heading_text(only_line)


def _chapter_title_looks_inferred_from_body(chapter: Chapter) -> bool:
    inferred = _title_from_text(chapter.text)
    if not inferred:
        return False
    return _normalize_heading_text(chapter.title) == _normalize_heading_text(inferred)


def _should_merge_title_only_chapter_with_next(
    chapter: Chapter, next_chapter: Chapter
) -> bool:
    if not _is_title_only_chapter(chapter):
        return False
    if len(next_chapter.text) < _TITLE_ONLY_MERGE_MIN_NEXT_TEXT_LEN:
        return False
    if not _chapter_title_looks_inferred_from_body(next_chapter):
        return False
    next_first = _chapter_first_nonempty_line(next_chapter.text)
    if not next_first:
        return False
    if _normalize_heading_text(chapter.title) == _normalize_heading_text(next_first):
        return False
    if _looks_like_short_structural_heading(next_first):
        return False
    return True


def _prepend_title_to_text(title: str, text: str) -> str:
    cleaned_title = _normalize_heading_text(title)
    if not cleaned_title:
        return text
    first_line = _chapter_first_nonempty_line(text)
    if _normalize_heading_text(first_line) == cleaned_title:
        return text
    return f"{cleaned_title}\n\n{text}"


def _merge_title_only_chapter_with_next(chapter: Chapter, next_chapter: Chapter) -> Chapter:
    chapter_headings = list(chapter.headings)
    chapter_heading_categories = dict(chapter.heading_categories)
    title_key = _normalize_heading_key(chapter.title)
    if title_key:
        if title_key not in {_normalize_heading_key(value) for value in chapter_headings}:
            chapter_headings.insert(0, chapter.title)
        chapter_heading_categories[title_key] = _merge_heading_category(
            chapter_heading_categories.get(title_key, _HEADING_CATEGORY_SECTION),
            _HEADING_CATEGORY_TITLE,
        )
    merged_headings, merged_categories = _merge_headings_with_categories(
        chapter_headings,
        chapter_heading_categories,
        next_chapter.headings,
        next_chapter.heading_categories,
    )
    merged_text = _prepend_title_to_text(chapter.title, next_chapter.text)
    return Chapter(
        title=chapter.title,
        href=next_chapter.href,
        source=next_chapter.source,
        text=merged_text,
        headings=merged_headings,
        heading_categories=merged_categories,
    )


def _merge_chapter_with_continuation(chapter: Chapter, continuation: Chapter) -> Chapter:
    left_text = chapter.text
    right_text = continuation.text
    if left_text and right_text:
        glue = "\n\n"
    else:
        glue = ""
    merged_text = f"{left_text}{glue}{right_text}"
    merged_headings, merged_categories = _merge_headings_with_categories(
        list(chapter.headings),
        dict(chapter.heading_categories),
        continuation.headings,
        continuation.heading_categories,
    )
    return Chapter(
        title=chapter.title,
        href=chapter.href,
        source=chapter.source,
        text=merged_text,
        headings=merged_headings,
        heading_categories=merged_categories,
    )


def _chapter_start_hrefs_from_title_overrides(title_overrides: dict[str, str]) -> set[str]:
    starts: set[str] = set()
    for href, title in title_overrides.items():
        cleaned = str(title or "").strip()
        if not cleaned:
            continue
        if _looks_like_filename(cleaned, href):
            continue
        starts.add(href)
    return starts


def _is_continuation_followup_item(
    item: object,
    footnote_index: dict[str, set[str]] | None,
    heading_class_markers: Optional[set[str]],
    heading_id_markers: Optional[set[str]],
) -> bool:
    content = item.get_content()
    if not content:
        return True
    headings, _ = _extract_html_headings_and_categories(
        content,
        heading_class_markers=heading_class_markers,
        heading_id_markers=heading_id_markers,
    )
    for heading in headings:
        if _looks_like_structural_title(heading):
            return False
    text = html_to_text(
        content,
        footnote_index=footnote_index,
        source_href=_item_name(item),
    )
    if not text.strip():
        return True
    first_line = _chapter_first_nonempty_line(text)
    if _looks_like_structural_title(first_line):
        return False
    return True


def _augment_toc_chapters_with_followups(
    chapters: List[Chapter],
    book: epub.EpubBook,
    toc_entries: Iterable[TocEntry],
    footnote_index: dict[str, set[str]] | None,
    heading_class_markers: Optional[set[str]],
    heading_id_markers: Optional[set[str]],
    extra_boundary_hrefs: Optional[set[str]] = None,
) -> List[Chapter]:
    spine_items = _build_spine_items(book)
    spine_index = {href: idx for idx, (href, _item) in enumerate(spine_items)}
    boundary_positions: set[int] = set()
    for entry in toc_entries:
        h = normalize_href(entry.href)
        spine_pos = spine_index.get(h)
        if spine_pos is not None:
            boundary_positions.add(spine_pos)
    for href in extra_boundary_hrefs or ():
        spine_pos = spine_index.get(href)
        if spine_pos is not None:
            boundary_positions.add(spine_pos)
    sorted_boundaries = sorted(boundary_positions)

    augmented: List[Chapter] = []
    for chapter in chapters:
        source = normalize_href(chapter.source)
        pos = spine_index.get(source)
        if pos is None:
            augmented.append(chapter)
            continue
        chapter_title = chapter.title or ""
        if (
            _is_non_content_toc_title(chapter_title)
            or _looks_like_filename(chapter_title, source)
        ):
            augmented.append(chapter)
            continue

        next_pos = next(
            (p for p in sorted_boundaries if p > pos), len(spine_items)
        )

        followup_items: List[object] = []
        for p in range(pos + 1, next_pos):
            _href, item = spine_items[p]
            if not _is_continuation_followup_item(
                item,
                footnote_index,
                heading_class_markers,
                heading_id_markers,
            ):
                break
            followup_items.append(item)

        if not followup_items:
            augmented.append(chapter)
            continue

        ftext = _join_item_text(followup_items, footnote_index=footnote_index)
        if not ftext:
            augmented.append(chapter)
            continue

        cur_text = chapter.text
        glue = "\n\n" if cur_text and ftext else ""
        new_text = cur_text + glue + ftext

        new_headings = list(chapter.headings)
        new_categories = dict(chapter.heading_categories)
        for fitem in followup_items:
            f_content = fitem.get_content()
            f_headings, f_categories = _extract_html_headings_and_categories(
                f_content,
                heading_class_markers=heading_class_markers,
                heading_id_markers=heading_id_markers,
            )
            new_headings, new_categories = _merge_headings_with_categories(
                new_headings, new_categories, f_headings, f_categories
            )

        augmented.append(
            Chapter(
                title=chapter.title,
                href=chapter.href,
                source=chapter.source,
                text=new_text,
                headings=new_headings,
                heading_categories=new_categories,
            )
        )

    return augmented


def _is_spine_continuation_fragment(chapter: Chapter) -> bool:
    if _looks_like_filename(chapter.title, chapter.source):
        return True
    if chapter.headings:
        return False
    if not _chapter_title_looks_inferred_from_body(chapter):
        return False
    if _looks_like_structural_title(chapter.title):
        return False
    return True


def _merge_spine_chapters_by_start_hrefs(
    chapters: List[Chapter], start_hrefs: set[str]
) -> List[Chapter]:
    if not chapters or not start_hrefs:
        return chapters
    merged: List[Chapter] = []
    current = chapters[0]
    for chapter in chapters[1:]:
        if chapter.source in start_hrefs:
            merged.append(current)
            current = chapter
            continue
        if _is_spine_continuation_fragment(chapter):
            current = _merge_chapter_with_continuation(current, chapter)
            continue
        merged.append(current)
        current = chapter
    merged.append(current)
    return merged


def _merge_title_only_chapters(chapters: List[Chapter]) -> List[Chapter]:
    merged: List[Chapter] = []
    idx = 0
    while idx < len(chapters):
        chapter = chapters[idx]
        if idx + 1 < len(chapters):
            next_chapter = chapters[idx + 1]
            if _should_merge_title_only_chapter_with_next(chapter, next_chapter):
                merged.append(_merge_title_only_chapter_with_next(chapter, next_chapter))
                idx += 2
                continue
        merged.append(chapter)
        idx += 1
    return merged


def _build_spine_title_overrides_from_toc(
    book: epub.EpubBook,
    entries: Iterable[TocEntry],
    footnote_index: dict[str, set[str]] | None = None,
) -> tuple[dict[str, str], set[str]]:
    overrides: dict[str, str] = {}
    for entry in entries:
        href = normalize_href(entry.href)
        if not href:
            continue
        title = str(entry.title or "").strip()
        if not title:
            continue
        overrides[href] = title
    if not overrides:
        return overrides, set()

    spine_items = _build_spine_items(book)
    if not spine_items:
        return overrides, set()
    spine_index = {href: idx for idx, (href, _item) in enumerate(spine_items)}
    spine_hrefs = set(spine_index)
    toc_hrefs = set(overrides.keys())
    inherited_hrefs: set[str] = set()
    text_cache: dict[str, str] = {}

    # Some omnibus EPUBs place chapter labels in body-level CONTENTS pages
    # instead of the package TOC. Harvest those link texts as additional title
    # candidates before running title-page -> body inheritance.
    for source_href, source_item in spine_items:
        content = source_item.get_content()
        if not content:
            continue
        soup = _parse_html_soup(content)
        root = soup.body if soup.body else soup
        links = root.find_all("a")
        if len(links) < _INLINE_TOC_MIN_LINKS:
            continue
        link_entries: List[tuple[str, str]] = []
        for link in links:
            href_attr = str(link.get("href", "") or "").strip()
            if not href_attr:
                continue
            resolved = _resolve_relative_href(source_href, href_attr)
            if not resolved or resolved not in spine_hrefs:
                continue
            label = _normalize_heading_text(link.get_text(separator=" ", strip=True))
            if not label or len(label) > 120:
                continue
            link_entries.append((resolved, label))
        if len(link_entries) < _INLINE_TOC_MIN_LINKS:
            continue
        for resolved, label in link_entries:
            existing = str(overrides.get(resolved, "")).strip()
            if existing and not (
                _looks_like_filename(existing, resolved)
                or _is_non_content_toc_title(existing)
            ):
                continue
            overrides[resolved] = label

    def item_text(href: str, item: object) -> str:
        if href in text_cache:
            return text_cache[href]
        text = html_to_text(
            item.get_content(),
            footnote_index=footnote_index,
            source_href=_item_name(item),
        )
        text_cache[href] = text
        return text

    for href, title in list(overrides.items()):
        if _looks_like_filename(title, href):
            continue
        if _is_non_content_toc_title(title):
            continue
        idx = spine_index.get(href)
        if idx is None:
            continue
        current_href, current_item = spine_items[idx]
        if item_text(current_href, current_item):
            continue
        for next_idx in range(idx + 1, len(spine_items)):
            next_href, next_item = spine_items[next_idx]
            if not item_text(next_href, next_item):
                continue
            if next_href in toc_hrefs:
                next_toc_title = str(overrides.get(next_href, "")).strip()
                if next_toc_title and not _looks_like_filename(next_toc_title, next_href):
                    break
            next_override = str(overrides.get(next_href, "")).strip()
            if next_override and not _looks_like_filename(next_override, next_href):
                break
            overrides[next_href] = title
            inherited_hrefs.add(next_href)
            break

    return overrides, inherited_hrefs


def _chapters_from_entries(
    book: epub.EpubBook,
    entries: Iterable[TocEntry],
    footnote_index: dict[str, set[str]] | None = None,
    title_overrides: Optional[dict[str, str]] = None,
    fallback_prefix: str = "Chapter",
    heading_class_markers: Optional[set[str]] = None,
    heading_id_markers: Optional[set[str]] = None,
    prepend_title_hrefs: Optional[set[str]] = None,
) -> List[Chapter]:
    seen: set[str] = set()
    chapters: List[Chapter] = []

    for idx, entry in enumerate(entries, start=1):
        base_href = normalize_href(entry.href)
        if not base_href or base_href in seen:
            continue
        seen.add(base_href)

        item = book.get_item_with_href(base_href)
        if not item:
            item = book.get_item_with_id(base_href)
        if not item or item.get_type() != ITEM_DOCUMENT:
            continue

        content = item.get_content()
        text = html_to_text(
            content,
            footnote_index=footnote_index,
            source_href=_item_name(item),
        )
        if not text:
            continue
        headings, heading_categories = _extract_html_headings_and_categories(
            content,
            heading_class_markers=heading_class_markers,
            heading_id_markers=heading_id_markers,
        )

        title = ""
        if title_overrides and base_href in title_overrides:
            title = str(title_overrides[base_href] or "").strip()
        if not title:
            title = entry.title or _item_title(item) or ""
        if _looks_like_filename(title, base_href):
            heading = _extract_heading_title(
                item.get_content(),
                heading_class_markers=heading_class_markers,
                heading_id_markers=heading_id_markers,
            )
            if heading and not _looks_like_filename(heading, base_href):
                title = heading
        if _looks_like_filename(title, base_href):
            text_title = _title_from_text(text)
            if text_title and not _looks_like_filename(text_title, base_href):
                title = text_title
        if _looks_like_filename(title, base_href):
            title = f"{fallback_prefix} {idx}"
        if prepend_title_hrefs and base_href in prepend_title_hrefs:
            text = _prepend_title_to_text(title, text)
        resolved_heading_categories = _heading_categories_for_values(
            headings, heading_categories
        )
        resolved_heading_categories = _apply_chapter_title_heading_category(
            title,
            headings,
            resolved_heading_categories,
        )
        chapters.append(
            Chapter(
                title=title,
                href=entry.href,
                source=base_href,
                text=text,
                headings=headings,
                heading_categories=resolved_heading_categories,
            )
        )

    return _merge_title_only_chapters(chapters)


def _chapters_from_toc_entries(
    book: epub.EpubBook,
    entries: Iterable[TocEntry],
    footnote_index: dict[str, set[str]] | None = None,
    heading_class_markers: Optional[set[str]] = None,
    heading_id_markers: Optional[set[str]] = None,
    apply_title_only_merge: bool = True,
) -> List[Chapter]:
    spine_items = _build_spine_items(book)
    spine_index = {href: idx for idx, (href, _item) in enumerate(spine_items)}
    chapters: List[Chapter] = []
    seen: set[str] = set()
    split_series_counts: dict[tuple[str, str], int] = {}
    for entry in entries:
        base_href = normalize_href(entry.href)
        if not base_href:
            continue
        key = _split_series_key(base_href)
        if key:
            split_series_counts[key] = split_series_counts.get(key, 0) + 1

    for idx, entry in enumerate(entries, start=1):
        base_href = normalize_href(entry.href)
        if not base_href or base_href in seen:
            continue
        seen.add(base_href)
        headings: List[str] = []
        heading_categories: dict[str, str] = {}

        start_idx = spine_index.get(base_href)
        merged_items: List[object] = []
        if start_idx is not None:
            key = _split_series_key(base_href)
            prev_key = (
                _split_series_key(spine_items[start_idx - 1][0])
                if start_idx > 0
                else None
            )
            if (
                key
                and key != prev_key
                and split_series_counts.get(key, 0) == 1
            ):
                cur = start_idx
                while cur < len(spine_items):
                    href, item = spine_items[cur]
                    if _split_series_key(href) != key:
                        break
                    merged_items.append(item)
                    cur += 1

        if merged_items:
            text = _join_item_text(merged_items, footnote_index=footnote_index)
            for merged in merged_items:
                merged_content = merged.get_content()
                merged_headings, merged_categories = _extract_html_headings_and_categories(
                    merged_content,
                    heading_class_markers=heading_class_markers,
                    heading_id_markers=heading_id_markers,
                )
                headings, heading_categories = _merge_headings_with_categories(
                    headings,
                    heading_categories,
                    merged_headings,
                    merged_categories,
                )
            item_for_title = merged_items[0]
        else:
            item_for_title = None
            if start_idx is not None:
                item_for_title = spine_items[start_idx][1]
            if not item_for_title:
                item_for_title = book.get_item_with_href(base_href)
            if not item_for_title:
                item_for_title = book.get_item_with_id(base_href)
            if not item_for_title or item_for_title.get_type() != ITEM_DOCUMENT:
                continue
            content = item_for_title.get_content()
            text = html_to_text(
                content,
                footnote_index=footnote_index,
                source_href=_item_name(item_for_title),
            )
            headings, heading_categories = _extract_html_headings_and_categories(
                content,
                heading_class_markers=heading_class_markers,
                heading_id_markers=heading_id_markers,
            )

        if not text:
            continue

        title = entry.title or _item_title(item_for_title) or Path(base_href).stem
        if _looks_like_filename(title, base_href):
            heading = _extract_heading_title(
                item_for_title.get_content(),
                heading_class_markers=heading_class_markers,
                heading_id_markers=heading_id_markers,
            )
            if heading and not _looks_like_filename(heading, base_href):
                title = heading
        if _looks_like_filename(title, base_href):
            text_title = _title_from_text(text)
            if text_title and not _looks_like_filename(text_title, base_href):
                title = text_title
        if _looks_like_filename(title, base_href):
            title = f"Chapter {idx}"
        resolved_heading_categories = _heading_categories_for_values(
            headings, heading_categories
        )
        resolved_heading_categories = _apply_chapter_title_heading_category(
            title,
            headings,
            resolved_heading_categories,
        )
        chapters.append(
            Chapter(
                title=title,
                href=entry.href,
                source=base_href,
                text=text,
                headings=headings,
                heading_categories=resolved_heading_categories,
            )
        )

    if apply_title_only_merge:
        return _merge_title_only_chapters(chapters)
    return chapters


def extract_chapters(book: epub.EpubBook, prefer_toc: bool = True) -> List[Chapter]:
    footnote_index = _collect_footnote_index(book)
    heading_class_markers, heading_id_markers = _collect_book_heading_markers(book)
    entries = build_toc_entries(book) if prefer_toc else []
    spine_title_overrides: dict[str, str] = {}
    spine_prepend_title_hrefs: set[str] = set()
    spine_start_hrefs: set[str] = set()
    if entries and prefer_toc:
        spine_title_overrides, spine_prepend_title_hrefs = (
            _build_spine_title_overrides_from_toc(
                book,
                entries,
                footnote_index=footnote_index,
            )
        )
        spine_start_hrefs = _chapter_start_hrefs_from_title_overrides(spine_title_overrides)

    def build_spine_chapters() -> List[Chapter]:
        title_overrides = spine_title_overrides if spine_title_overrides else None
        prepend_hrefs = spine_prepend_title_hrefs if spine_prepend_title_hrefs else None
        chapters_out = _chapters_from_entries(
            book,
            build_spine_entries(book),
            footnote_index,
            title_overrides=title_overrides,
            heading_class_markers=heading_class_markers,
            heading_id_markers=heading_id_markers,
            prepend_title_hrefs=prepend_hrefs,
        )
        if spine_start_hrefs:
            chapters_out = _merge_spine_chapters_by_start_hrefs(
                chapters_out, spine_start_hrefs
            )
        return chapters_out
    chapters = (
        _chapters_from_toc_entries(
            book,
            entries,
            footnote_index,
            heading_class_markers=heading_class_markers,
            heading_id_markers=heading_id_markers,
            apply_title_only_merge=False,
        )
        if entries
        else []
    )
    if entries and not chapters:
        chapters = _chapters_from_entries(
            book,
            entries,
            footnote_index,
            heading_class_markers=heading_class_markers,
            heading_id_markers=heading_id_markers,
        )
    used_spine = False
    if chapters and prefer_toc:
        spine_chapters = build_spine_chapters()
        if spine_chapters:
            toc_chars = sum(len(ch.text) for ch in chapters if ch.text)
            spine_chars = sum(len(ch.text) for ch in spine_chapters if ch.text)
            if spine_chars > 0:
                ratio = toc_chars / spine_chars
                if ratio < _TOC_SPINE_COVERAGE_MIN_RATIO:
                    chapters = spine_chapters
                    used_spine = True
    if not chapters:
        chapters = build_spine_chapters()
        used_spine = True
    if entries and prefer_toc and not used_spine:
        chapters = _augment_toc_chapters_with_followups(
            chapters,
            book,
            entries,
            footnote_index,
            heading_class_markers,
            heading_id_markers,
            extra_boundary_hrefs=spine_start_hrefs,
        )
        chapters = _merge_title_only_chapters(chapters)
    return chapters
