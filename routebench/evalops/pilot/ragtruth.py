"""RAGTruth summarisation adapter: responses + span labels -> sentence-level source-support examples.

DECLARED LABEL POLICY (version `ragtruth-sentence-v1`; every rule below is enforced and tested with synthetic fixtures):

  Target meaning: label 1 = the sentence is supported by the supplied source document; 0 = it is not.
  Only `task_type == "Summary"` records are adapted. Sources are restricted to `allowed_sources`
  (default {"CNN/DM"}); "Recent News" is EXCLUDED PROVISIONALLY because no permission evidence for those articles was found.
  Nothing is ever labelled supported by default: a record that cannot be interpreted is REJECTED with a reason.

  Record checks (a failure rejects the whole response, never a single label):
    * required fields present with the right types; `split` in {train, test}; `labels` must be a list (a missing or null
      `labels` is rejected, it is NOT read as "no hallucination")
    * `quality` must be "good"; "truncated" and "incorrect_refusal" are excluded with their own reason; any other value is
      rejected as unknown
    * `source_id` must resolve to exactly one source record whose `task_type` is "Summary" and whose `source_info` is text
      (a dict or other type is rejected)
  Span checks:
    * integer offsets with 0 <= start < end <= len(response); by default offsets are half-open (`exclusive`) and
      `response[start:end]` must equal the span's `text` exactly, otherwise the response is rejected (`offset_mismatch`).
      `probe_offset_convention` reports which convention a sample of real records satisfies, to be run once after data
      access is approved; the convention is then fixed, never guessed per record.
    * `label_type` must be one of the known RAGTruth types (conflict / baseless information, evident / subtle)
    * `due_to_null` is a data-to-text artefact; on a Summary record it is rejected as unexpected
    * `implicit_true` spans (unsupported by the source but true in the world) are UNSUPPORTED for this target, because the
      target is support by the supplied document; the flag is kept on the example. The alternative policy `exclude` drops
      those sentences instead. Neither policy ever marks them supported.
    * a span lying entirely in the whitespace between sentences is ambiguous and rejects the response
    * overlapping spans are merged; a span that crosses a sentence boundary marks EVERY sentence it touches unsupported
  Sentence rule: deterministic offset-preserving splitter (`SENTENCE_SPLITTER_VERSION`, regex-v2: title, initial and
  abbreviation aware; quotation aware; single newlines are line wraps unless they follow terminal punctuation or precede a list
  marker; blank lines always separate); a sentence is unsupported iff a merged span intersects it, else supported.
  Known splitter limitations (magnitude NOT measured): an abbreviation followed by a capitalised non-starter word ("in the
  U.S. Officials said") is not split, and any heuristic can mis-split other text. A merged sentence inherits "unsupported" if any
  span touches it, which can label a supported neighbour unsupported; the size of that effect on real data is unknown until the
  splitter is audited against the approved data.
  Non-content wrappers (text-only since v4): a leading interjection or colon-terminated "Here is/are ..." preface (at most the
  first two segments; list prefaces only when they carry no digits or names) is out of sentence-level scope and listed in
  `AdapterResult.wrappers`. Segmentation and scope depend on the text alone, never on annotations. A span that touches a wrapper
  is reported (`Wrapper.unsupported`, `stats["wrapper_with_span"]`) and still makes the RESPONSE unsupported. A response with no
  content sentences is rejected. Whitespace-like HTML entities are treated as spaces by the splitter without decoding the text.
  Response aggregation (a PROPOSED rule, not a reproduction of FRANK's human policy): response supported iff every segment,
  wrappers included, is supported. The non-default `implicit_true_policy="exclude"` is annotation-dependent and must not be used
  for validation or inference inputs.
  Partitioning: every response about the same source document (by normalised text fingerprint, not by `source_id`) lands in
  one partition; a document that appears under both original `train` and `test` splits is excluded from training and reported.
  Original train/test document membership is recorded from EVERY response with a resolvable Summary source and a valid split,
  before any quality, label or offset filtering, so a rejected test response cannot erase its document's test membership.
  Duplicate response ids reject every record sharing the id (`duplicate_response_id`); none is kept or overwritten.
  Original `id`, `source_id` and `split` are preserved on every example.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

LABEL_POLICY_VERSION = "ragtruth-sentence-v4"
SENTENCE_SPLITTER_VERSION = "regex-v5"
KNOWN_LABEL_TYPES = frozenset({
    "evident conflict", "subtle conflict", "evident baseless info", "subtle baseless info",
    "evident introduction of baseless information", "subtle introduction of baseless information",
})
VALID_SPLITS = frozenset({"train", "test"})
DEFAULT_ALLOWED_SOURCES = frozenset({"CNN/DM"})


def normalized_fingerprint(text: str) -> str:
    """Case- and punctuation-insensitive document fingerprint (alphanumerics only)."""
    return hashlib.sha256(re.sub(r"[^A-Za-z0-9]+", "", text).lower().encode()).hexdigest()


_CLOSERS = "\"')]”’"
_TERMINAL = re.compile(r"([.!?]+)([" + re.escape(_CLOSERS) + r"]*)(\s+)")
_NEWLINE = re.compile(r"[ \t]*\n(\s*)")
_OPENERS = "\"'([“‘"
_LIST_MARKER = re.compile(r"(?:[-*•]\s|\d+[.)]\s)")
#: A period after one of these never ends a sentence (they precede a name or number).
TITLES = frozenset({"dr", "mr", "mrs", "ms", "prof", "st", "mt", "gen", "sen", "rep", "gov", "lt", "col", "capt", "sgt",
                    "rev", "hon", "messrs", "mme", "fig", "vol", "no"})
#: Abbreviations that usually continue the sentence, but end it when the next word is a common sentence starter.
ABBREVIATIONS = frozenset({"u.s", "u.k", "u.n", "e.u", "a.m", "p.m", "e.g", "i.e", "etc", "vs", "inc", "ltd", "co", "corp",
                           "jr", "sr", "dept", "est", "approx", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
                           "sept", "oct", "nov", "dec", "ph.d", "b.a", "m.a"})
SENTENCE_STARTERS = frozenset({"the", "he", "she", "it", "they", "we", "i", "you", "in", "on", "at", "but", "and", "a", "an",
                               "this", "that", "these", "those", "there", "however", "then", "after", "before", "when",
                               "while", "his", "her", "their", "its", "our", "if", "as", "for", "with", "some", "many"})
_LIST_NUMBER = re.compile(r"(?:^|\n)[ \t]*\d{1,2}$")
_WORD_BEFORE = re.compile(r"([A-Za-z]+(?:\.[A-Za-z]+)*)$")
_NEXT_WORD = re.compile(r"[A-Za-z]+")


def _inside_open_quote(text: str, pos: int) -> bool:
    """True when `pos` lies inside a double quotation that has been opened but not yet closed."""
    before = text[:pos]
    return before.count('"') % 2 == 1 or before.count("“") > before.count("”")


_SPACE_ENTITY = re.compile(r"&(?:nbsp|ensp|emsp|thinsp|#160|#xa0|#8203);", re.IGNORECASE)


def _period_continues_sentence(text: str, dot_start: int, next_start: int, closers: str = "") -> bool:
    """True when the single period at `dot_start` belongs to a title, initial or abbreviation rather than ending a sentence."""
    before = text[:dot_start]
    if _LIST_NUMBER.search(before):  # "1." / "12." at the start of a line is a list marker, not a sentence end
        return True
    m = _WORD_BEFORE.search(before)
    if not m:
        return False
    token = m.group(1)
    low = token.lower()
    if low in TITLES:
        return True
    if len(token) == 1 and token.isupper():  # an initial: "J. K. Rowling"
        return True
    if low in ABBREVIATIONS or "." in token:  # any dotted token ("A.D", "R.J", "Ph.D") is abbreviation-like
        # Inside an unclosed quotation an abbreviation period does not end the sentence, unless the quotation closes right after
        # the period (`... the U.S." She left.`), in which case the normal starter test decides.
        if _inside_open_quote(text, dot_start) and not any(c in closers for c in "\"\u201d"):
            return True
        nxt = _NEXT_WORD.match(text, next_start)
        return not (nxt and nxt.group(0).lower() in SENTENCE_STARTERS)
    return False


_END_OF_SENTENCE = re.compile(r"([.!?]+)([" + re.escape(_CLOSERS) + r"]*)$")


def _newline_follows_sentence_end(text: str, before: str, next_start: int) -> bool:
    """True when the text before a single newline ends in terminal punctuation that really ends a sentence.

    A closing quote or bracket alone is not a sentence end, and a period after a title, initial or abbreviation is judged by
    the same rules as on a single line, using the first word on the next line.
    """
    if before.endswith(":"):  # a colon-terminated line introduces the block that follows
        return True
    m = _END_OF_SENTENCE.search(before)
    if not m:
        return False
    return not (m.group(1) == "." and _period_continues_sentence(text, m.start(1), next_start, m.group(2)))


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Half-open (start, end) spans of non-whitespace sentence text, in order, offsets into `text` (splitter regex-v5).

    A boundary is terminal punctuation (with optional closing quotes/brackets) followed by whitespace and a token that looks like
    a sentence start (capital, digit, opening quote/bracket), unless the period belongs to a title, initial or abbreviation.
    Blank lines always separate; a single newline separates only after terminal punctuation or before a list marker (otherwise it
    is treated as line wrapping inside one sentence).
    """
    # Whitespace-like HTML entities (for example "&nbsp;") are blanked to spaces of the SAME length, so boundary detection sees
    # whitespace while every offset still indexes the original, undecoded text.
    text = _SPACE_ENTITY.sub(lambda m: " " * len(m.group()), text)
    cuts: set[tuple[int, int]] = set()
    for m in _TERMINAL.finditer(text):
        nxt_at = m.end()
        if nxt_at >= len(text):
            continue
        nxt = text[nxt_at]
        if not (nxt.isupper() or nxt.isdigit() or nxt in _OPENERS):
            continue
        if m.group(1) == "." and _period_continues_sentence(text, m.start(1), nxt_at, m.group(2)):
            continue
        cuts.add((m.end(2), nxt_at))
    for m in _NEWLINE.finditer(text):
        nxt_at = m.end()
        if nxt_at >= len(text):
            continue
        before = text[:m.start()].rstrip()
        blank_line = "\n" in m.group(1)
        if blank_line or _newline_follows_sentence_end(text, before, nxt_at) or _LIST_MARKER.match(text, nxt_at):
            cuts.add((len(before), nxt_at))
    spans: list[tuple[int, int]] = []
    cursor = 0
    for end, nxt in sorted(cuts):
        if nxt > cursor:
            spans.append((cursor, end))
            cursor = nxt
    spans.append((cursor, len(text)))
    out = []
    for s0, e0 in spans:
        while s0 < e0 and text[s0].isspace():
            s0 += 1
        while e0 > s0 and text[e0 - 1].isspace():
            e0 -= 1
        if e0 > s0:
            out.append((s0, e0))
    return out


_INTERJECTION = re.compile(r"^(?:sure|certainly|of course|okay|ok|absolutely|alright)\s*[!.,]?$", re.IGNORECASE)
_PREFACE_START = re.compile(r"^(?:(?:sure|certainly|of course|okay|ok)[,!]?\s+)?(?:here(?:'s| is| are)|below (?:is|are)|"
                            r"the following (?:is|are))\b", re.IGNORECASE)
_PREFACE_NOUN = re.compile(r"\b(?:summary|summaries|main points|key points|highlights|overview|recap)\b", re.IGNORECASE)


def wrapper_reason(sentence: str) -> str | None:
    """Reason a leading segment is a non-content wrapper (interjection or summary preface), else None.

    Deliberately narrow: an interjection on its own, or a short colon-terminated "Here is/are ..." preface that announces what
    follows (a summary, or a list). A segment ending in a colon announces content rather than asserting it. Anything else that
    could carry a source claim is left alone.
    """
    s = sentence.strip()
    if _INTERJECTION.match(s):
        return "interjection"
    if s.endswith(":") and len(s.split()) <= 30 and _PREFACE_START.match(s):
        if _PREFACE_NOUN.search(s):
            return "summary_preface"
        # A bare list preface must carry no specific content: short, no digits, no capitalised name after the first word.
        words = s.split()
        if len(words) <= 12 and not any(ch.isdigit() for ch in s) and not any(w[:1].isupper() for w in words[1:]):
            return "list_preface"
    return None


@dataclass(frozen=True)
class Wrapper:
    response_id: str
    sentence_index: int
    reason: str
    sentence: str
    unsupported: bool = False  # gold-derived, reported only: it never influences segmentation or scope


@dataclass(frozen=True)
class Rejection:
    response_id: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class SentenceExample:
    example_id: str
    response_id: str
    source_id: str
    original_split: str
    partition: str
    document: str
    document_fingerprint: str
    sentence: str
    sentence_index: int
    start: int
    end: int
    label: int
    implicit_true: bool = False
    crosses_boundary: bool = False


@dataclass
class AdapterResult:
    examples: list[SentenceExample] = field(default_factory=list)
    response_labels: dict[str, int] = field(default_factory=dict)
    rejected: list[Rejection] = field(default_factory=list)
    wrappers: list[Wrapper] = field(default_factory=list)
    cross_split_documents: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


def _reject(out: AdapterResult, rid: str, reason: str, detail: str = "") -> None:
    out.rejected.append(Rejection(rid, reason, detail))
    out.stats[f"rejected:{reason}"] = out.stats.get(f"rejected:{reason}", 0) + 1


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if merged and s < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _check_span(span: Any, response: str, convention: str) -> tuple[int, int, bool, str] | str:
    """Return (start, end, implicit_true, label_type) with half-open offsets, or a rejection reason string."""
    if not isinstance(span, Mapping):
        return "malformed_label"
    start, end, text = span.get("start"), span.get("end"), span.get("text")
    if not (isinstance(start, int) and isinstance(end, int) and not isinstance(start, bool)
            and not isinstance(end, bool) and isinstance(text, str)):
        return "malformed_label"
    end_excl = end if convention == "exclusive" else end + 1
    if not 0 <= start < end_excl <= len(response):
        return "offset_out_of_range"
    if response[start:end_excl] != text:
        return "offset_mismatch"
    ltype = span.get("label_type")
    if not isinstance(ltype, str) or ltype.strip().lower() not in KNOWN_LABEL_TYPES:
        return "unknown_label_type"
    if span.get("due_to_null"):
        return "unexpected_due_to_null"
    implicit = span.get("implicit_true", False)
    if not isinstance(implicit, bool):
        return "malformed_label"
    return start, end_excl, implicit, ltype.strip().lower()


def probe_offset_convention(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count how many labelled spans match each offset convention (run once on approved data; decide once)."""
    counts = {"exclusive": 0, "inclusive": 0, "both": 0, "neither": 0}
    for r in records:
        resp = r.get("response")
        for span in r.get("labels") or []:
            if not (isinstance(resp, str) and isinstance(span, Mapping)):
                continue
            s, e, t = span.get("start"), span.get("end"), span.get("text")
            if not (isinstance(s, int) and isinstance(e, int) and isinstance(t, str)):
                continue
            ex, inc = resp[s:e] == t, resp[s:e + 1] == t
            counts["both" if ex and inc else "exclusive" if ex else "inclusive" if inc else "neither"] += 1
    return counts


def adapt(
    responses: Iterable[Mapping[str, Any]],
    sources: Iterable[Mapping[str, Any]],
    *,
    allowed_sources: frozenset[str] = DEFAULT_ALLOWED_SOURCES,
    implicit_true_policy: str = "unsupported",
    offset_convention: str = "exclusive",
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> AdapterResult:
    """Join, validate and convert. Returns examples plus an explicit rejection list; nothing is dropped silently."""
    if implicit_true_policy not in ("unsupported", "exclude"):
        raise ValueError("implicit_true_policy must be 'unsupported' or 'exclude'")
    if offset_convention not in ("exclusive", "inclusive"):
        raise ValueError("offset_convention must be 'exclusive' or 'inclusive'")
    out = AdapterResult()
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for s in sources:
        by_source[str(s.get("source_id"))].append(s)

    responses = list(responses)
    id_counts: dict[str, int] = defaultdict(int)
    for r in responses:
        id_counts[str(r.get("id"))] += 1

    # Original train/test document membership is recorded BEFORE any quality, label or offset filtering, so a response that is
    # rejected later (for example a truncated test response) cannot erase its document's membership in that split.
    doc_splits: dict[str, set[str]] = defaultdict(set)
    for r in responses:
        src_rows = by_source.get(str(r.get("source_id")), [])
        if (len(src_rows) == 1 and r.get("split") in VALID_SPLITS and src_rows[0].get("task_type") == "Summary"
                and isinstance(src_rows[0].get("source_info"), str) and src_rows[0]["source_info"].strip()):
            doc_splits[normalized_fingerprint(src_rows[0]["source_info"])].add(r["split"])
    out.cross_split_documents = sorted(fp for fp, sp in doc_splits.items() if len(sp) > 1)
    cross = set(out.cross_split_documents)

    accepted: list[tuple[Mapping[str, Any], str]] = []
    for r in responses:
        rid = str(r.get("id"))
        if id_counts[rid] > 1:
            _reject(out, rid, "duplicate_response_id", f"{id_counts[rid]} records share this id")
            continue
        if not all(isinstance(r.get(k), str) and r.get(k) for k in ("id", "source_id", "response")):
            _reject(out, rid, "malformed_record")
            continue
        if r.get("split") not in VALID_SPLITS:
            _reject(out, rid, "invalid_split", str(r.get("split")))
            continue
        if not isinstance(r.get("labels"), list):
            _reject(out, rid, "missing_labels")
            continue
        src = by_source.get(r["source_id"], [])
        if len(src) != 1:
            _reject(out, rid, "source_join_failed", f"{len(src)} matches")
            continue
        s = src[0]
        if s.get("task_type") != "Summary":
            _reject(out, rid, "not_summary_task", str(s.get("task_type")))
            continue
        if s.get("source") not in allowed_sources:
            _reject(out, rid, "source_not_allowed", str(s.get("source")))
            continue
        if not isinstance(s.get("source_info"), str) or not s["source_info"].strip():
            _reject(out, rid, "source_info_not_text")
            continue
        quality = r.get("quality")
        if quality != "good":
            _reject(out, rid, f"quality_{quality}" if quality in ("truncated", "incorrect_refusal") else "unknown_quality",
                    str(quality))
            continue
        accepted.append((r, s["source_info"]))

    for r, doc in accepted:
        rid, resp = r["id"], r["response"]
        fp = normalized_fingerprint(doc)
        if fp in cross:
            _reject(out, rid, "document_in_train_and_test", fp[:12])
            continue
        checked = [_check_span(sp, resp, offset_convention) for sp in r["labels"]]
        bad = next((c for c in checked if isinstance(c, str)), None)
        if bad is not None:
            _reject(out, rid, bad)
            continue
        good = [c for c in checked if not isinstance(c, str)]
        sents = sentence_spans(resp)
        if not sents:
            _reject(out, rid, "no_sentences")
            continue
        merged = _merge([(s, e) for s, e, _, _ in good])
        if any(not any(s < se and e > ss for ss, se in sents) for s, e in merged):
            _reject(out, rid, "span_in_gap")
            continue
        implicit_spans = [(s, e) for s, e, imp, _ in good if imp]
        # Input construction is TEXT-ONLY: the same response text yields the same segments and the same scope whatever the
        # annotations say. A leading non-content wrapper ("Sure!", "Here is a summary of the article in 50 words:") is out of
        # sentence-level scope. Annotations are applied afterwards; a span on a wrapper is reported and still makes the
        # RESPONSE unsupported, so annotated content is never silently discarded.
        wrapper_at: dict[int, str] = {}
        for i, (ss, se) in enumerate(sents[:2]):
            reason = wrapper_reason(resp[ss:se])
            if reason is None:
                break
            wrapper_at[i] = reason
        rows: list[SentenceExample] = []
        response_supported = True
        skip_response = False
        for i, (ss, se) in enumerate(sents):
            hit = [(s, e) for s, e in merged if s < se and e > ss]
            if i in wrapper_at:
                touched = bool(hit)
                response_supported = response_supported and not touched
                out.wrappers.append(Wrapper(rid, i, wrapper_at[i], resp[ss:se], unsupported=touched))
                out.stats["wrapper_sentences"] = out.stats.get("wrapper_sentences", 0) + 1
                if touched:
                    out.stats["wrapper_with_span"] = out.stats.get("wrapper_with_span", 0) + 1
                continue
            unsupported = bool(hit)
            implicit = any(s < se and e > ss for s, e in implicit_spans)
            if unsupported and implicit and implicit_true_policy == "exclude":
                skip_response = True
                break
            crosses = any(s < ss or e > se for s, e in hit)
            rows.append(SentenceExample(
                example_id=f"{rid}#{i}", response_id=rid, source_id=r["source_id"], original_split=r["split"],
                partition="", document=doc, document_fingerprint=fp, sentence=resp[ss:se], sentence_index=i,
                start=ss, end=se, label=0 if unsupported else 1, implicit_true=implicit, crosses_boundary=crosses))
        if skip_response:
            _reject(out, rid, "implicit_true_excluded")
            continue
        if not rows:
            _reject(out, rid, "no_content_sentences")
            continue
        out.examples.extend(rows)
        out.response_labels[rid] = int(response_supported and all(x.label == 1 for x in rows))

    out.examples = assign_partitions(out.examples, validation_fraction=validation_fraction, seed=seed)
    out.stats["examples"] = len(out.examples)
    out.stats["responses"] = len(out.response_labels)
    out.stats["rejected"] = len(out.rejected)
    return out


def assign_partitions(examples: list[SentenceExample], *, validation_fraction: float, seed: int) -> list[SentenceExample]:
    """train/validation for original-train documents, `test` for original-test ones. Grouped by document fingerprint."""
    def bucket(fp: str) -> float:
        return int(hashlib.sha256(f"{seed}:{fp}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    result = []
    for x in examples:
        if x.original_split == "test":
            part = "test"
        else:
            part = "validation" if bucket(x.document_fingerprint) < validation_fraction else "train"
        result.append(SentenceExample(**{**x.__dict__, "partition": part}))
    groups: dict[str, set[str]] = defaultdict(set)
    for x in result:
        groups[x.document_fingerprint].add(x.partition)
    assert all(len(v) == 1 for v in groups.values()), "a document was split across partitions"
    return result
