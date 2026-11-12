#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pymupdf>=1.24"]
# ///
"""Prepare a local-paper workspace from a PDF.

Run with uv so the local PDF fallback dependency is resolved automatically, with
no system packages and no virtualenv to manage:

    uv run scripts/prepare_local_pdf.py paper.pdf out-dir/

By default (`--backend auto`), the script prefers `mineru-open-api extract`
when the CLI is installed and authenticated, producing structured Markdown
(`paper.md` with physical page markers mapped from `paper.json`) plus cropped
figure/chart/table images in `images/`.

When `mineru-open-api` is unavailable, unauthenticated, fails, or when
`--backend pymupdf` is requested (e.g. offline or confidential drafts), the
script falls back to PyMuPDF (`paper.txt` + full-page renders `page-NNN.png`),
and finally to poppler if run without uv/PyMuPDF.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Embedded images below this pixel size are decorative (rules, logos, glyph
# fragments) and only add noise to the index. Filter on dimensions only, never
# on encoded byte size: line-art figures and plots compress to very few bytes.
MIN_IMAGE_PIXELS = 120
PAGE_RENDER_DPI = 200
DEFAULT_MINERU_TIMEOUT = 300


def load_pymupdf():
    """Return the pymupdf module, or None when it is unavailable."""
    try:
        import pymupdf  # type: ignore

        return pymupdf
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore  # PyMuPDF < 1.24 name

            return pymupdf
        except ImportError:
            return None


def which(name: str) -> str | None:
    return shutil.which(name)


def run(
    cmd: list[str],
    *,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# --- MinerU OpenAPI CLI path -----------------------------------------------


def check_mineru_ready(cli: str, token: str | None) -> tuple[bool, str]:
    """Check whether mineru-open-api has a configured or supplied token."""
    if token or os.environ.get("MINERU_TOKEN", "").strip():
        return True, "token provided via flag/env"
    verify = run([cli, "auth", "--verify"], timeout=15)
    if verify.returncode == 0:
        return True, "authenticated via mineru-open-api config"
    detail = (verify.stderr or verify.stdout).strip().splitlines()
    reason = detail[0] if detail else "token missing"
    return False, reason


def inject_page_markers(raw_md: str, blocks: list[dict]) -> tuple[str, int]:
    """Insert HTML comment page markers into MinerU Markdown using paper.json.

    Markers take the form `<!-- ===== [page N] ===== -->` at paragraph boundaries
    so they stay invisible in Markdown previews while keeping PDF page numbers
    searchable and citable.
    """
    if not raw_md.strip() or not blocks:
        return raw_md, 0

    pages_present = sorted(
        {
            b["page_idx"]
            for b in blocks
            if isinstance(b, dict) and isinstance(b.get("page_idx"), int)
        }
    )
    if not pages_present:
        return raw_md, 0

    insertions: list[tuple[int, int]] = []
    search_pos = 0

    for page_idx in pages_present:
        page_blocks = [
            b
            for b in blocks
            if isinstance(b, dict) and b.get("page_idx") == page_idx
        ]
        # Running headers and page footnotes are often omitted or moved in .md;
        # try body blocks first, then any remaining blocks on that page.
        primary = [
            b for b in page_blocks if b.get("type") not in ("header", "page_footnote")
        ]
        candidates = primary + [
            b for b in page_blocks if b.get("type") in ("header", "page_footnote")
        ]

        matched_offset: int | None = None
        for block in candidates:
            img_path = block.get("img_path")
            raw_cand = (
                block.get("text")
                or block.get("table_body")
                or (f"![]({img_path})" if img_path else "")
            )
            cand = (raw_cand or "").strip()
            if not cand:
                continue
            probe = cand.splitlines()[0][:60].strip()
            if not probe:
                continue
            found = raw_md.find(probe, search_pos)
            if found != -1:
                para_sep = raw_md.rfind("\n\n", 0, found)
                matched_offset = 0 if para_sep == -1 else para_sep + 2
                search_pos = max(search_pos, found)
                break

        if matched_offset is not None:
            insertions.append((matched_offset, page_idx + 1))

    if not insertions:
        return raw_md, 0

    # Rebuild string with page markers inserted in order.
    pieces: list[str] = []
    cursor = 0
    for offset, page_num in insertions:
        offset = max(offset, cursor)
        pieces.append(raw_md[cursor:offset])
        marker = f"<!-- ===== [page {page_num}] ===== -->\n\n"
        if offset > 0 and pieces[-1] and not pieces[-1].endswith("\n\n"):
            if not pieces[-1].endswith("\n"):
                pieces.append("\n\n")
            else:
                pieces.append("\n")
        pieces.append(marker)
        cursor = offset
    pieces.append(raw_md[cursor:])
    annotated = "".join(pieces).rstrip() + "\n"
    return annotated, len(insertions)


def clean_caption(values: object) -> str:
    if isinstance(values, list):
        joined = " ".join(str(v).strip() for v in values if str(v).strip())
    elif isinstance(values, str):
        joined = values.strip()
    else:
        joined = ""
    joined = " ".join(joined.split())
    if len(joined) > 160:
        return joined[:157] + "..."
    return joined


def build_mineru_image_entries(
    images_dir: Path,
    blocks: list[dict],
    raw_md: str,
) -> tuple[list[str], int, int]:
    """Build structured index entries from MinerU JSON metadata and image files."""
    by_name: dict[str, dict] = {}
    for block in blocks:
        if not isinstance(block, dict):
            continue
        img_path = block.get("img_path")
        if not isinstance(img_path, str) or not img_path:
            continue
        name = Path(img_path).name
        by_name.setdefault(name, block)

    all_files = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.name != "index.md"
    )
    primary_entries: list[str] = []
    aux_entries: list[str] = []

    for img_file in all_files:
        name = img_file.name
        block = by_name.get(name)
        in_md = name in raw_md
        if block is not None:
            page_no = int(block.get("page_idx", 0)) + 1
            btype = str(block.get("type", "image"))
            caption = (
                clean_caption(block.get("image_caption"))
                or clean_caption(block.get("chart_caption"))
                or clean_caption(block.get("table_caption"))
            )
            link_note = "linked in `paper.md`" if in_md else f"{btype} crop"
            desc = f"- `{name}`: page {page_no}, {btype} ({link_note})"
            if caption:
                desc += f" — {caption}"
            primary_entries.append(desc)
        elif in_md:
            primary_entries.append(f"- `{name}`: linked in `paper.md`")
        else:
            aux_entries.append(f"- `{name}`: auxiliary/formula crop")

    entries: list[str] = []
    if primary_entries:
        entries.extend(primary_entries)
    if aux_entries:
        if entries:
            entries.append("")
            entries.append("## Auxiliary / Formula Crops")
            entries.append("")
        entries.extend(aux_entries)
    return entries, len(primary_entries), len(aux_entries)


def extract_with_mineru(
    cli: str,
    copied_pdf: Path,
    output_dir: Path,
    images_dir: Path,
    *,
    use_ocr: bool,
    language: str | None,
    timeout: int,
    token: str | None,
) -> tuple[bool, str, str, list[str]]:
    """Run mineru-open-api extract into a temp staging dir and normalize outputs."""
    ver_proc = run([cli, "version"], timeout=10)
    cli_version = (ver_proc.stdout or ver_proc.stderr).strip().splitlines()
    version_label = cli_version[0] if cli_version else "mineru-open-api"

    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".mineru_tmp_") as tmp:
        stage_dir = Path(tmp)
        cmd = [
            cli,
            "extract",
            str(copied_pdf),
            "-f",
            "md,json",
            "--timeout",
            str(timeout),
            "-o",
            str(stage_dir),
        ]
        if use_ocr:
            cmd.append("--ocr")
        if language:
            cmd.extend(["-l", language])
        if token:
            cmd.extend(["--token", token])

        try:
            proc = run(cmd, timeout=timeout + 60)
        except subprocess.TimeoutExpired:
            return (
                False,
                f"mineru-open-api timed out after {timeout + 60}s",
                "",
                [],
            )

        staged_md = stage_dir / f"{copied_pdf.stem}.md"
        staged_json = stage_dir / f"{copied_pdf.stem}.json"
        if proc.returncode != 0 or not staged_md.exists():
            err = (proc.stderr or proc.stdout).strip()
            err_line = err.splitlines()[-1] if err else f"exit {proc.returncode}"
            return False, f"mineru-open-api failed ({err_line})", "", []

        raw_md = staged_md.read_text(encoding="utf-8")
        if not raw_md.strip():
            return False, "mineru-open-api produced an empty Markdown file", "", []

        blocks: list[dict] = []
        if staged_json.exists():
            try:
                loaded = json.loads(staged_json.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    blocks = loaded
            except json.JSONDecodeError:
                blocks = []
            shutil.copy2(staged_json, output_dir / "paper.json")

        annotated_md, marked_pages = inject_page_markers(raw_md, blocks)
        write_text(output_dir / "paper.md", annotated_md)

        staged_images = stage_dir / "images"
        if staged_images.is_dir():
            for item in sorted(staged_images.iterdir()):
                if item.is_file():
                    shutil.copy2(item, images_dir / item.name)

        entries, primary_count, aux_count = build_mineru_image_entries(
            images_dir, blocks, raw_md
        )

    ocr_flag = "ocr=on" if use_ocr else "ocr=off"
    backend_line = f"backend: {version_label} (extract -f md,json, {ocr_flag})"
    text_status = (
        f"text: extracted to paper.md with mineru-open-api "
        f"({marked_pages} page markers mapped from paper.json)"
    )
    image_status = (
        f"images: mineru-open-api extracted {primary_count} figure/chart/table crops"
    )
    if aux_count:
        image_status += f" + {aux_count} auxiliary/formula crops"
    return True, backend_line, f"{text_status}\n{image_status}", entries


# --- PyMuPDF path ----------------------------------------------------------


def extract_text_pymupdf(doc, output_txt: Path) -> str:
    """Write the full text with page markers, so claims stay citable by page."""
    chunks = []
    for number, page in enumerate(doc, start=1):
        chunks.append(f"\n\n===== [page {number}] =====\n\n")
        chunks.append(page.get_text("text"))
    text = "".join(chunks).strip()
    if not text:
        return "text: pymupdf found no text layer (likely a scanned PDF; OCR needed)"
    write_text(output_txt, text + "\n")
    return f"text: extracted with pymupdf ({len(doc)} pages, page markers included)"


def render_pages_pymupdf(doc, images_dir: Path, dpi: int) -> list[str]:
    entries = []
    for number, page in enumerate(doc, start=1):
        name = f"page-{number:03d}.png"
        page.get_pixmap(dpi=dpi).save(images_dir / name)
        entries.append(f"- `{name}`: rendered page {number} @ {dpi}dpi")
    return entries


def extract_images_pymupdf(
    pymupdf, doc, images_dir: Path, dpi: int
) -> tuple[str, list[str]]:
    """Render every page, then add whatever embedded rasters are actually usable.

    Page renders are the reliable artifact when MinerU is unavailable. Embedded
    image extraction is only a supplement, because in a typical CS paper the
    figures are vector art: the objects `get_images()` reports are then just the
    soft masks and shading layers behind that art, and `extract_image()` hands
    back a base layer that opens as solid black. Such images are composited with
    their mask here, and dropped when the result carries no detail.
    """
    entries = render_pages_pymupdf(doc, images_dir, dpi)
    page_count = len(entries)

    seen: set[int] = set()
    kept = 0
    skipped_flat = 0

    for number, page in enumerate(doc, start=1):
        for info in page.get_images(full=True):
            xref, smask_xref = info[0], info[1]
            if xref in seen:  # the same image can be placed on several pages
                continue
            seen.add(xref)
            try:
                pix = pymupdf.Pixmap(doc, xref)
                if smask_xref:  # re-attach transparency, else the art reads as black
                    pix = pymupdf.Pixmap(pix, pymupdf.Pixmap(doc, smask_xref))
            except Exception:  # noqa: BLE001 - a broken xref must not abort the run
                continue
            if pix.width < MIN_IMAGE_PIXELS or pix.height < MIN_IMAGE_PIXELS:
                continue
            if pix.is_unicolor:  # a bare mask or a solid fill: no figure in there
                skipped_flat += 1
                continue
            name = f"img-{kept:03d}.png"
            pix.save(images_dir / name)
            entries.append(
                f"- `{name}`: page {number}, {pix.width}x{pix.height}px (embedded)"
            )
            kept += 1

    status = (
        f"images: rendered {page_count} pages @ {dpi}dpi + kept {kept} embedded rasters"
    )
    if skipped_flat:
        status += f" (dropped {skipped_flat} mask-only/flat images)"
    return status, entries


# --- poppler fallback ------------------------------------------------------


def extract_text_poppler(pdf_path: Path, output_txt: Path) -> str:
    pdftotext = which("pdftotext")
    if not pdftotext:
        return (
            "text: no extractor available (install uv and run this script with `uv run`)"
        )
    result = run([pdftotext, str(pdf_path), "-"])
    if result.returncode == 0 and result.stdout.strip():
        write_text(output_txt, result.stdout)
        return "text: extracted with pdftotext (fallback; no page markers)"
    return f"text: pdftotext failed ({result.stderr.strip() or 'no stderr'})"


def extract_images_poppler(pdf_path: Path, images_dir: Path) -> str:
    pdfimages = which("pdfimages")
    if pdfimages:
        result = run([pdfimages, "-all", str(pdf_path), str(images_dir / "img")])
        if result.returncode == 0:
            return "images: extracted with pdfimages (fallback)"
        return f"images: pdfimages failed ({result.stderr.strip() or 'no stderr'})"

    pdftoppm = which("pdftoppm")
    if pdftoppm:
        result = run([pdftoppm, "-png", str(pdf_path), str(images_dir / "page")])
        if result.returncode == 0:
            return "images: rendered pages with pdftoppm (fallback)"
        return f"images: pdftoppm failed ({result.stderr.strip() or 'no stderr'})"

    return (
        "images: no extractor available (install uv and run this script with `uv run`)"
    )


def build_index_from_dir(images_dir: Path) -> list[str]:
    files = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.name != "index.md"
    )
    return [f"- `{item.name}`" for item in files]


def write_index(images_dir: Path, entries: list[str]) -> None:
    lines = ["# Image Index", ""]
    if entries:
        lines.extend(entries)
        lines.extend(
            [
                "",
                "为每张计划使用的图补上：对应图号（Fig./Table）、建议用途、是否进正文。",
            ]
        )
    else:
        lines.append("- No extracted images or rendered pages were created.")
    write_text(images_dir / "index.md", "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a local-paper workspace from a PDF (run via `uv run`).",
    )
    parser.add_argument("pdf", help="Path to the source PDF")
    parser.add_argument("output_dir", help="Directory to populate")
    parser.add_argument(
        "--backend",
        choices=("auto", "mineru", "pymupdf"),
        default="auto",
        help="Extraction backend: prefer mineru-open-api ('auto'), force 'mineru', or force local 'pymupdf' (default: auto)",
    )
    parser.add_argument(
        "--no-mineru-ocr",
        action="store_true",
        help="Disable --ocr when calling mineru-open-api extract (enabled by default to prevent PDF font CMap glyph corruption)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code passed to mineru-open-api extract (-l), e.g. 'en' or 'ch'",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_MINERU_TIMEOUT,
        help=f"Timeout in seconds for mineru-open-api extract (default: {DEFAULT_MINERU_TIMEOUT})",
    )
    parser.add_argument(
        "--mineru-token",
        default=None,
        help="Optional MinerU API token override (prefer `mineru-open-api auth` or MINERU_TOKEN env var)",
    )
    parser.add_argument(
        "--render-pages",
        action="store_true",
        help="Also render full PDF pages (`page-NNN.png`) via PyMuPDF even when MinerU succeeds",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=PAGE_RENDER_DPI,
        help=f"Page render resolution for PyMuPDF (default: {PAGE_RENDER_DPI})",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not pdf_path.exists():
        print(f"error: PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    copied_pdf = output_dir / "source.pdf"
    if pdf_path != copied_pdf:
        shutil.copy2(pdf_path, copied_pdf)

    report_lines = [f"source_pdf: {copied_pdf}"]
    used_mineru = False
    entries: list[str] = []

    if args.backend in ("auto", "mineru"):
        mineru_cli = which("mineru-open-api")
        if not mineru_cli:
            msg = "mineru_note: mineru-open-api not found in PATH"
            if args.backend == "mineru":
                print(f"error: {msg}", file=sys.stderr)
                return 1
            report_lines.append(f"{msg}; falling back to local extractor")
        else:
            ready, auth_reason = check_mineru_ready(mineru_cli, args.mineru_token)
            if not ready:
                msg = f"mineru_note: mineru-open-api unavailable ({auth_reason})"
                if args.backend == "mineru":
                    print(f"error: {msg}", file=sys.stderr)
                    return 1
                report_lines.append(f"{msg}; falling back to local extractor")
            else:
                ok, backend_or_err, status_block, entries = extract_with_mineru(
                    mineru_cli,
                    copied_pdf,
                    output_dir,
                    images_dir,
                    use_ocr=not args.no_mineru_ocr,
                    language=args.language,
                    timeout=args.timeout,
                    token=args.mineru_token,
                )
                if ok:
                    used_mineru = True
                    report_lines.append(backend_or_err)
                    report_lines.append(status_block)
                else:
                    if args.backend == "mineru":
                        print(f"error: {backend_or_err}", file=sys.stderr)
                        return 1
                    report_lines.append(
                        f"mineru_note: {backend_or_err}; falling back to local extractor"
                    )

    if used_mineru and args.render_pages:
        pymupdf = load_pymupdf()
        if pymupdf is not None:
            with pymupdf.open(copied_pdf) as doc:
                page_entries = render_pages_pymupdf(doc, images_dir, args.dpi)
            if page_entries:
                entries.extend(["", "## Full-Page Renders", "", *page_entries])
                report_lines.append(
                    f"page_renders: rendered {len(page_entries)} pages @ {args.dpi}dpi via pymupdf"
                )
        else:
            report_lines.append(
                "page_renders: requested --render-pages, but PyMuPDF is unavailable"
            )

    if not used_mineru:
        pymupdf = load_pymupdf()
        if pymupdf is not None:
            with pymupdf.open(copied_pdf) as doc:
                backend = f"backend: pymupdf {pymupdf.__doc__ or ''}".strip()
                text_status = extract_text_pymupdf(doc, output_dir / "paper.txt")
                image_status, entries = extract_images_pymupdf(
                    pymupdf, doc, images_dir, args.dpi
                )
        else:
            backend = "backend: poppler fallback (PyMuPDF unavailable — prefer `uv run`)"
            text_status = extract_text_poppler(copied_pdf, output_dir / "paper.txt")
            image_status = extract_images_poppler(copied_pdf, images_dir)
            entries = build_index_from_dir(images_dir)
        report_lines.extend([backend, text_status, image_status])

    write_index(images_dir, entries)

    report = "\n".join(report_lines)
    write_text(output_dir / "prep_status.txt", report + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
