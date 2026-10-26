"""Shared infrastructure for annotator review packs (German Excel workbooks).

A review pack is an .xlsx file the annotator fills in by hand and returns.
This module holds what every pack needs, independent of its questions:

  - Excel-safe text handling and formula-injection protection;
  - row fingerprints, so the reader detects edited texts and IDs;
  - writing answer sheets with a hidden machine-key row, drop-down lists,
    unlocked answer cells, a protected instruction sheet and a hidden Meta
    sheet;
  - saving a workbook exclusively (never overwriting) with a fixed timestamp;
  - reading answer sheets back by machine key, mapping drop-down labels to
    codes and checking IDs;
  - hashing files and writing JSON exclusively, for artifact directories
    that are created new on every run.

The questions themselves (columns, option lists, instructions) live in the
pack module, e.g. reaction_split_review.py.
"""
from __future__ import annotations

import hashlib
import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill, Protection
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.writer.excel import ExcelWriter

CREATED_UTC_FORMAT = "%Y%m%dT%H%M%S%fZ"
HASHES_FILE = "artifact_hashes.json"
EXCEL_CELL_LIMIT = 32767
KEY_ROW, HEADER_ROW, FIRST_DATA_ROW = 1, 2, 3

FONT = Font(name="Arial", size=10)
BOLD = Font(name="Arial", size=10, bold=True)
TITLE = Font(name="Arial", size=12, bold=True)
ANSWER_FILL = PatternFill("solid", fgColor="FFFF99")
HEADER_FILL = PatternFill("solid", fgColor="D9D9D9")
WRAP = Alignment(wrap_text=True, vertical="top")
UNLOCKED = Protection(locked=False)
# openpyxl stores a string starting with "=" as a formula, and Excel may evaluate cells starting
# with any of these characters (formula injection); such texts are forced to plain text in write_cell.
_FORMULA_LEADS = ("=", "+", "-", "@")

OptionSets = Mapping[str, Sequence[tuple[str, str]]]


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes, read in 64 KiB chunks.

    Args:
        path: File to hash.

    Returns:
        The lower-case hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        # iter(callable, sentinel) keeps calling f.read until it returns b"" (end of file).
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping) -> None:
    """Write pretty-printed UTF-8 JSON to a new file.

    Args:
        path: Target path; must not exist yet (mode "x").
        payload: JSON-serializable mapping; key order is preserved.

    Raises:
        FileExistsError: If path already exists.
    """
    with open(path, "x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")


def write_artifact_hashes(output_dir: Path, created_utc: str, files: Sequence[Path]) -> Path:
    """Write artifact_hashes.json, the completion marker of an artifact directory.

    It must be written last: a directory without it is an incomplete run.

    Args:
        output_dir: The artifact directory.
        created_utc: Creation timestamp in CREATED_UTC_FORMAT.
        files: The artifact files to hash (by name).

    Returns:
        The path of the written file.

    Raises:
        FileExistsError: If the file already exists.
    """
    path = output_dir / HASHES_FILE
    write_json(path, {"created_utc": created_utc, "complete": True,
                      "files": {file.name: sha256_file(file) for file in files}})
    return path


# --------------------------------------------------------------------------
# Texts
# --------------------------------------------------------------------------


def clean(value: object) -> object:
    """Turn missing scalars into None.

    Args:
        value: Any cell value.

    Returns:
        None for None, NaN and pd.NA; the value unchanged otherwise.
    """
    # pd.isna on a list/array returns an array, so it is only applied to scalars;
    # list cells pass through unchanged.
    return None if pd.api.types.is_scalar(value) and pd.isna(value) else value


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip the ends.

    Args:
        text: Any text.

    Returns:
        The text with single spaces.
    """
    return " ".join(text.split())


def safe_text(value: object) -> tuple[str | None, int]:
    """Make a text Excel-safe.

    Newlines are normalized to "\\n" and the text is stripped, characters
    Excel cannot store are replaced with U+FFFD, and the result is capped at
    EXCEL_CELL_LIMIT characters.

    Args:
        value: Any value (stringified); None stays None.

    Returns:
        A tuple of the safe text (None for None) and the number of replaced
        illegal characters.
    """
    if value is None:
        return None, 0
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    # ILLEGAL_CHARACTERS_RE: the control characters openpyxl refuses to write (XML 1.0 cannot hold them).
    text, replaced = ILLEGAL_CHARACTERS_RE.subn("\ufffd", text)
    return text[:EXCEL_CELL_LIMIT], replaced


def display(value: object) -> str | None:
    """Display form of a stored text: HTML entities decoded and Excel-safe.

    Args:
        value: A stored text, or a missing value.

    Returns:
        The display text, or None when the value is missing.
    """
    value = clean(value)
    return None if value is None else safe_text(html.unescape(str(value)))[0]


def row_fingerprint(*parts: str) -> str:
    """16-hex fingerprint of a row's identifying cells, as written.

    The reader recomputes it from the cells it reads back, so any edit to a
    text or ID cell of that row is detected.

    Args:
        *parts: The identifying cell values, in a fixed order.

    Returns:
        The first 16 hex characters of SHA-256 over the joined parts.
    """
    # "\x1f" (ASCII unit separator) cannot occur in Excel-safe texts (safe_text replaces it) or hex IDs,
    # so the join is unambiguous: ("ab", "c") and ("a", "bc") give different fingerprints.
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnSpec:
    """One pack column: machine key (row 1), German header (row 2), layout and editability.

    Attributes:
        key: Machine key written into the hidden key row; the reader locates
            columns by it, so columns may be reordered.
        header: German header shown in the header row.
        width: Excel column width.
        hidden: Whether the column is hidden (IDs and fingerprints).
            Defaults to False.
        editable: Whether the annotator fills the cell (unlocked, coloured,
            read back by the reader). Defaults to False.
        list_name: Option-set key of the drop-down list, or None for free
            text. Defaults to None.
    """

    key: str
    header: str
    width: float
    hidden: bool = False
    editable: bool = False
    list_name: str | None = None


def write_cell(ws, row: int, column: int, value: object, font: Font = FONT) -> None:
    """Write one value through safe_text, with font and wrapping.

    Missing scalars (None, NaN, pd.NA) leave the cell empty, and strings
    that look like formulas are stored as plain text.

    Args:
        ws: The worksheet.
        row: 1-based row.
        column: 1-based column.
        value: The value to write.
        font: Cell font. Defaults to FONT.
    """
    value = clean(value)
    if isinstance(value, str):
        value = safe_text(value)[0]
    cell = ws.cell(row=row, column=column, value=value)
    if isinstance(value, str) and value.startswith(_FORMULA_LEADS):
        # openpyxl infers data_type "f" (formula) for strings starting with "="; forcing "s" writes the
        # text as a string, and quotePrefix keeps Excel from re-parsing it as a formula when edited.
        cell.data_type = "s"
        cell.quotePrefix = True
    cell.font, cell.alignment = font, WRAP


def list_validation(formula: str) -> DataValidation:
    """A drop-down list validation that rejects values not in the list.

    Args:
        formula: Range formula of the allowed values (from write_lists).

    Returns:
        The DataValidation; empty cells are allowed.
    """
    return DataValidation(type="list", formula1=formula, allow_blank=True, showErrorMessage=True,
                          errorTitle="Ungültiger Wert", error="Bitte einen Wert aus der Liste wählen.")


def write_lists(ws, option_sets: OptionSets) -> dict[str, str]:
    """Write the option labels to a lists sheet and hide it.

    Each option set gets one column: its name in row 1 and the German labels
    below.

    Args:
        ws: The lists worksheet.
        option_sets: Option-set name to (code, German label) pairs.

    Returns:
        Option-set name to an absolute range formula such as
        "=Listen!$A$2:$A$7", for use in data validations.
    """
    ranges = {}
    for column, (name, options) in enumerate(option_sets.items(), start=1):
        letter = get_column_letter(column)
        ws.cell(row=1, column=column, value=name).font = BOLD
        for row, (_, label) in enumerate(options, start=2):
            ws.cell(row=row, column=column, value=label).font = FONT
        ranges[name] = f"='{ws.title}'!${letter}$2:${letter}${len(options) + 1}"
    ws.sheet_state = "hidden"
    return ranges


def write_table_sheet(ws, specs: Sequence[ColumnSpec], rows: list[dict], list_ranges: Mapping[str, str],
                      *, freeze_column: str = "C") -> None:
    """Write one answer sheet.

    Row KEY_ROW holds the machine keys (hidden), HEADER_ROW the German
    headers, data starts at FIRST_DATA_ROW. Editable cells are unlocked,
    coloured and get a drop-down list when the column has one. The sheet is
    left unprotected, so the annotator can sort, filter and resize; the
    reader detects edited texts and IDs by row fingerprints instead.

    Args:
        ws: The worksheet to fill.
        specs: The sheet's columns, in order.
        rows: Row dicts keyed by ColumnSpec.key.
        list_ranges: Option-set name to its range formula (from write_lists).
        freeze_column: First column right of the frozen panes. Defaults to
            "C" (keeps the first two columns visible while scrolling).
    """
    for column, spec in enumerate(specs, start=1):
        ws.cell(row=KEY_ROW, column=column, value=spec.key).font = FONT
        header = ws.cell(row=HEADER_ROW, column=column, value=spec.header or None)
        header.font, header.alignment, header.fill = BOLD, WRAP, HEADER_FILL
        dimension = ws.column_dimensions[get_column_letter(column)]
        dimension.width = spec.width
        dimension.hidden = spec.hidden
        for offset, values in enumerate(rows):
            row = FIRST_DATA_ROW + offset
            write_cell(ws, row, column, values.get(spec.key))
            if spec.editable:
                cell = ws.cell(row=row, column=column)
                cell.protection, cell.fill = UNLOCKED, ANSWER_FILL
        if spec.list_name and rows:
            letter = get_column_letter(column)
            validation = list_validation(list_ranges[spec.list_name])
            ws.add_data_validation(validation)
            validation.add(f"{letter}{FIRST_DATA_ROW}:{letter}{FIRST_DATA_ROW + len(rows) - 1}")
    ws.row_dimensions[KEY_ROW].hidden = True
    ws.freeze_panes = f"{freeze_column}{FIRST_DATA_ROW}"
    last = get_column_letter(len(specs))
    ws.auto_filter.ref = f"A{HEADER_ROW}:{last}{HEADER_ROW + max(len(rows), 1)}"


def write_text_sheet(ws, entries: Sequence[tuple[str, str]], legend_fill: PatternFill = ANSWER_FILL) -> None:
    """Write an instruction sheet of (kind, text) entries, then protect it.

    Kinds: "title", "heading" (column A, bold, preceded by an empty row),
    "text" (column B) and "answer_legend" (column B, with column A coloured
    like the answer cells).

    Args:
        ws: The worksheet.
        entries: The entries in display order.
        legend_fill: Fill of the legend swatch. Defaults to ANSWER_FILL.
    """
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 30, 110
    row = 1
    for kind, text in entries:
        if kind == "heading":
            row += 1
        font = {"title": TITLE, "heading": BOLD}.get(kind, FONT)
        write_cell(ws, row, 1 if kind in ("title", "heading") else 2, text, font)
        if kind == "answer_legend":
            ws.cell(row=row, column=1).fill = legend_fill
        row += 1
    # Protected without a password: the text cannot be edited by accident.
    ws.protection.sheet = True


def write_meta(ws, entries: Sequence[tuple[str, str]]) -> None:
    """Write key/value pairs to a Meta sheet and make it very hidden.

    Args:
        ws: The Meta worksheet.
        entries: (key, value) pairs.
    """
    for row, (key, value) in enumerate(entries, start=1):
        write_cell(ws, row, 1, key)
        write_cell(ws, row, 2, value)
    # "veryHidden" sheets cannot be unhidden from Excel's UI (only via VBA), unlike "hidden".
    ws.sheet_state = "veryHidden"


def save_workbook_exclusive(workbook: Workbook, path: Path, *, created_utc: str, title: str) -> None:
    """Save a workbook to a new file with fixed document properties.

    The created/modified properties are set to created_utc, so the file
    content does not depend on the wall clock.

    Args:
        workbook: The workbook to save.
        path: Target .xlsx path; must not exist yet.
        created_utc: Creation timestamp in CREATED_UTC_FORMAT.
        title: Document title property.

    Raises:
        FileExistsError: If path already exists.
    """
    created = datetime.strptime(created_utc, CREATED_UTC_FORMAT)
    properties = workbook.properties
    properties.creator = properties.lastModifiedBy = "stance-detection-app"
    properties.created = properties.modified = created
    properties.title = title
    # Mode "xb" (exclusive create) fails if the file exists; Workbook.save() would silently overwrite it.
    with open(path, "xb") as f, ZipFile(f, "w", ZIP_DEFLATED, allowZip64=True) as archive:
        # ExcelWriter directly: openpyxl's save() would overwrite `modified` with the wall clock.
        # write_data() leaves closing the archive to the "with" block.
        ExcelWriter(workbook, archive).write_data()


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def cell_text(value: object) -> str | None:
    """A read cell value as text with "\\n" newlines (as safe_text wrote them).

    Args:
        value: The cell value (numbers are stringified).

    Returns:
        The text, or None for an empty cell.
    """
    if value is None:
        return None
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def label_to_code(option_sets: OptionSets, list_name: str, value: object, where: str) -> str | None:
    """Map a drop-down label back to its option code.

    Args:
        option_sets: Option-set name to (code, German label) pairs.
        list_name: The option-set key.
        value: The cell value.
        where: Location used in the error message.

    Returns:
        The code, or None for an empty or blank cell.

    Raises:
        ValueError: If the value is not one of the list's labels.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    codes = {label: code for code, label in option_sets[list_name]}
    code = codes.get(str(value).strip())
    if code is None:
        raise ValueError(f"{where}: unknown value {value!r}")
    return code


def read_meta(workbook, meta_sheet: str, schema_version: str, required: Sequence[str]) -> dict[str, str]:
    """Read and check a Meta sheet.

    Args:
        workbook: The loaded pack.
        meta_sheet: Name of the Meta sheet.
        schema_version: Required value of the "pack_schema_version" key.
        required: Keys that must have a non-empty value.

    Returns:
        Key to value (empty cells as "").

    Raises:
        ValueError: If the Meta sheet is missing, the schema version differs
            or a required key is empty.
    """
    if meta_sheet not in workbook.sheetnames:
        raise ValueError("Not a review pack: Meta sheet missing")
    meta = {}
    for key, value in workbook[meta_sheet].iter_rows(min_col=1, max_col=2, values_only=True):
        if key is not None:
            meta[str(key)] = "" if value is None else str(value)
    if meta.get("pack_schema_version") != schema_version:
        raise ValueError(f"Unsupported pack schema version {meta.get('pack_schema_version')!r}; "
                         f"expected {schema_version!r}")
    missing = [key for key in required if not meta.get(key)]
    if missing:
        raise ValueError(f"Meta sheet lacks {missing}")
    return meta


def read_rows(ws, specs: Sequence[ColumnSpec], required: Sequence[str]) -> list[dict]:
    """Read the data rows of an answer sheet, locating columns by their machine key.

    Because columns are found via the hidden key row, the annotator may
    reorder columns and rows; fully empty rows are skipped.

    Args:
        ws: The worksheet.
        specs: The sheet's ColumnSpecs; only their keys are read.
        required: Keys that must be present.

    Returns:
        One dict per non-empty data row, keyed by the spec keys found.

    Raises:
        ValueError: If a key appears twice or a required key is missing.
    """
    positions = {}
    for index, cell in enumerate(ws[KEY_ROW]):
        if cell.value is None:
            continue
        if cell.value in positions:
            raise ValueError(f"{ws.title}: column key {cell.value!r} appears twice")
        positions[cell.value] = index
    missing = [key for key in required if key not in positions]
    if missing:
        raise ValueError(f"{ws.title}: missing columns {missing}")
    wanted = [spec.key for spec in specs if spec.key in positions]
    rows = []
    for values in ws.iter_rows(min_row=FIRST_DATA_ROW, values_only=True):
        # Defensive: a value tuple shorter than the key row yields None for the cells beyond its end.
        record = {key: cell_text(values[positions[key]]) if positions[key] < len(values) else None for key in wanted}
        if any(v is not None and v.strip() for v in record.values()):
            rows.append(record)
    return rows


def decisions(record: dict, specs: Sequence[ColumnSpec], sheet: str, option_sets: OptionSets) -> dict:
    """The editable cells of one row as answer values.

    Args:
        record: A row from read_rows.
        specs: The sheet's ColumnSpecs.
        sheet: Sheet name, used in error messages.
        option_sets: Option-set name to (code, German label) pairs.

    Returns:
        Per editable column key: the option code for list columns, the
        stripped text for free-text columns, None when empty.

    Raises:
        ValueError: If a list cell holds a value not in its list.
    """
    decided = {}
    for spec in specs:
        if not spec.editable:
            continue
        value, where = record.get(spec.key), f"{sheet} row {record.get('nr')}, column {spec.header!r}"
        decided[spec.key] = (label_to_code(option_sets, spec.list_name, value, where) if spec.list_name
                             else (value.strip() or None) if isinstance(value, str) else None)
    return decided


def check_ids(rows: list[dict], id_key: str, expected: Sequence[str], sheet: str) -> None:
    """Check that a sheet holds exactly the expected IDs, each once.

    Args:
        rows: Rows from read_rows.
        id_key: The ID column key.
        expected: The IDs recorded on the Meta sheet.
        sheet: Sheet name, used in error messages.

    Raises:
        ValueError: If a row lacks its ID, an ID repeats, or the set of IDs
            differs from the expected set.
    """
    seen = set()
    for record in rows:
        identifier = record.get(id_key)
        if not identifier:
            raise ValueError(f"{sheet} row {record.get('nr')}: missing {id_key}")
        if identifier in seen:
            raise ValueError(f"{sheet} row {record.get('nr')}: duplicate {id_key} {identifier}")
        seen.add(identifier)
    if seen != set(expected):
        absent, extra = sorted(set(expected) - seen), sorted(seen - set(expected))
        raise ValueError(f"{sheet}: rows missing for {absent} or unexpected rows {extra}")


def answer_frame(records: list[dict]) -> pd.DataFrame:
    """Build an object-dtype frame, so unanswered cells stay None instead of becoming NaN.

    Args:
        records: Parsed answer rows.

    Returns:
        The frame with a fresh 0-based index.
    """
    return pd.DataFrame(records, dtype=object).reset_index(drop=True)
