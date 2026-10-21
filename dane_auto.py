# -*- coding: utf-8 -*-
"""Descarga automática de archivos SIPSA-P del catálogo oficial del DANE.

La función intenta primero las solicitudes de formularios HTML asociadas al archivo
solicitado y después las rutas de descarga conocidas de NADA. Cada respuesta solo se
acepta si contiene un ZIP íntegro. Nunca se generan datos sintéticos.
"""
from __future__ import annotations

import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

DANE_CATALOG_URL = "https://microdatos.dane.gov.co/index.php/catalog/776/get-microdata"
DANE_FILE_PREFIX = "BaseDatos-SIPSA_P-Mensual-"
USER_AGENT = "AgroClima-SIPSA research app/1.1 (automated public data retrieval)"
MAX_ARCHIVE_BYTES = 1_500_000_000


@dataclass
class DownloadCandidate:
    method: str
    url: str
    data: dict[str, str] = field(default_factory=dict)
    label: str = ""


def archive_name_for_year(year: int) -> str:
    year = int(year)
    if 2013 <= year <= 2017:
        return f"{DANE_FILE_PREFIX}2013_2017.zip"
    if 2018 <= year <= 2024:
        return f"{DANE_FILE_PREFIX}{year}.zip"
    raise ValueError(
        "La ficha SIPSA-P consultada presenta archivos 2013–2024; "
        f"año fuera de rango: {year}."
    )


def _normal_url(candidate: str | None, base_url: str) -> str | None:
    candidate = (candidate or "").strip().strip("\"' ")
    if not candidate or candidate.startswith(("#", "javascript:", "mailto:")):
        return None
    candidate = candidate.replace("&amp;", "&")
    if candidate.startswith("//"):
        candidate = "https:" + candidate
    elif not candidate.startswith(("http://", "https://")):
        candidate = urljoin(base_url, candidate)
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc and not (
        parsed.netloc.endswith("dane.gov.co")
        or parsed.netloc.endswith("microdatos.dane.gov.co")
    ):
        return None
    return candidate


def _file_tokens(target_filename: str) -> set[str]:
    stem = Path(target_filename).stem
    without_prefix = stem[len("BaseDatos-"):] if stem.lower().startswith("basedatos-") else stem
    raw = {target_filename, stem, without_prefix}
    tokens = set()
    for item in raw:
        item = unquote(item).lower()
        tokens.add(item)
        tokens.add(re.sub(r"[^a-z0-9]", "", item))
    return {x for x in tokens if x}


def _file_id_from_catalog(html: str, target_filename: str) -> str | None:
    """Discover a NADA data-file identifier (e.g. F3) from its data-dictionary link."""
    soup = BeautifulSoup(html, "html.parser")
    tokens = _file_tokens(target_filename)
    for anchor in soup.find_all("a", href=True):
        href = unquote(str(anchor.get("href", "")))
        match = re.search(
            r"/catalog/\d+/(?:data-dictionary|data_dictionary|datafile|data-file)/([A-Za-z0-9_-]+)",
            href,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        values = [href, anchor.get_text(" ", strip=True)]
        query = parse_qs(urlparse(href).query)
        for vals in query.values():
            values.extend(vals)
        for parent in (anchor.parent, getattr(anchor.parent, "parent", None)):
            if parent is not None:
                values.append(parent.get_text(" ", strip=True)[:2000])
        context = " ".join(values).lower()
        compact = re.sub(r"[^a-z0-9]", "", context)
        if any(token in context or token in compact for token in tokens):
            return match.group(1)
    return None


def _discover_form_candidates(html: str, target_filename: str, base_url: str) -> list[DownloadCandidate]:
    """Inspect HTML forms that mention the requested file; preserve hidden/CSRF fields."""
    soup = BeautifulSoup(html, "html.parser")
    tokens = _file_tokens(target_filename)
    candidates: list[DownloadCandidate] = []
    seen = set()

    def has_target(context: str) -> bool:
        lower = unquote(context).lower()
        compact = re.sub(r"[^a-z0-9]", "", lower)
        return any(token in lower or token in compact for token in tokens)

    for form in soup.find_all("form"):
        values: dict[str, str] = {}
        context_parts = [form.get_text(" ", strip=True)]
        for attr in ("action", "id", "name", "class", "data-url", "data-action"):
            context_parts.append(str(form.attrs.get(attr, "")))
        # Preserve all fields, including CSRF fields required by some catalogue versions.
        for control in form.find_all(["input", "button", "select", "textarea"]):
            name = control.get("name")
            value = control.get("value")
            ctype = str(control.get("type", "")).lower()
            context_parts.extend([
                str(name or ""), str(value or ""), str(control.get("id", "")),
                str(control.get("onclick", "")),
            ])
            if name and ctype not in {"file", "image"}:
                if ctype in {"checkbox", "radio"} and not control.has_attr("checked"):
                    continue
                if value is not None:
                    values[str(name)] = str(value)
                elif control.name == "textarea":
                    values[str(name)] = control.get_text()
                elif control.name == "select":
                    selected = control.find("option", selected=True) or control.find("option")
                    if selected is not None:
                        values[str(name)] = str(selected.get("value", selected.get_text(strip=True)))
        context = " ".join(context_parts)
        if not has_target(context):
            continue

        # Do not submit a broad form if it visibly contains multiple yearly archives
        # and does not identify the requested file in its submitted values.
        archives = set(re.findall(
            r"(?:BaseDatos-)?SIPSA[_ -]?P-Mensual-(?:2013[_-]2017|20\d{2})(?:\.zip)?",
            context,
            flags=re.IGNORECASE,
        ))
        submitted_context = " ".join(f"{k}={v}" for k, v in values.items())
        if len(archives) > 1 and not has_target(submitted_context):
            continue

        action = _normal_url(
            str(form.get("action") or base_url), base_url
        )
        if not action:
            continue
        method = str(form.get("method", "GET")).upper()
        if method not in {"GET", "POST"}:
            continue
        key = (method, action, tuple(sorted(values.items())))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(DownloadCandidate(
            method=method,
            url=action,
            data=values,
            label=f"formulario HTML ({method}) {action}",
        ))
    return candidates


def _discover_get_candidates(html: str, target_filename: str, base_url: str) -> list[DownloadCandidate]:
    soup = BeautifulSoup(html, "html.parser")
    scored: dict[str, int] = {}

    def add(raw_url: str | None, score: int) -> None:
        url = _normal_url(raw_url, base_url)
        if url:
            scored[url] = max(score, scored.get(url, -1))

    file_id = _file_id_from_catalog(html, target_filename)
    if file_id:
        # Known NADA route variants. They are only accepted if the payload validates as ZIP.
        add(f"/index.php/catalog/776/download/{file_id}", 1000)
        add(f"/index.php/catalog/776/download/FILE/{file_id}", 1000)
        add(f"/index.php/catalog/776/download/FILE?file_id={quote(file_id)}", 950)
        add(f"/index.php/catalog/776/download?file_id={quote(file_id)}", 900)

    lower_target = target_filename.lower()
    for tag in soup.find_all(True):
        for attr in (
            "href", "src", "action", "formaction", "data-href", "data-url",
            "data-download-url", "onclick", "value",
        ):
            value = tag.attrs.get(attr)
            if not isinstance(value, str):
                continue
            if lower_target in value.lower() or (
                "download" in value.lower() and has_target_in_context(tag, target_filename)
            ):
                extracted = re.findall(
                    r"(?:https?://[^\s\"'<>]+|(?:/|\.\.?/)[^\s\"'<>]+)", value
                )
                for raw in (extracted if extracted else [value]):
                    add(raw, 500 if lower_target in raw.lower() else 100)

    escaped = quote(target_filename)
    escaped_stem = quote(Path(target_filename).stem)
    for raw in [
        f"/index.php/catalog/776/download/FILE?file_name={escaped}",
        f"/index.php/catalog/776/download?file_name={escaped}",
        f"/index.php/catalog/776/download/FILE/{escaped}",
        f"/index.php/catalog/776/download/{escaped}",
        f"/index.php/catalog/776/download/FILE?file_name={escaped_stem}",
        f"/index.php/catalog/776/get-microdata?file_name={escaped}",
    ]:
        add(raw, 1)
    return [
        DownloadCandidate(method="GET", url=url, label=f"GET {url}")
        for url, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)
    ]


def has_target_in_context(tag, target_filename: str) -> bool:
    """Check the nearby HTML container to avoid accepting unrelated download links."""
    tokens = _file_tokens(target_filename)
    node = tag
    for _ in range(4):
        if node is None:
            break
        text = node.get_text(" ", strip=True) if hasattr(node, "get_text") else ""
        attrs = " ".join(str(v) for v in getattr(node, "attrs", {}).values())
        context = unquote(f"{text} {attrs}").lower()
        compact = re.sub(r"[^a-z0-9]", "", context)
        if any(token in context or token in compact for token in tokens):
            return True
        node = getattr(node, "parent", None)
    return False


def _request_and_validate(
    session: requests.Session,
    candidate: DownloadCandidate,
    destination: Path,
    timeout: int = 120,
) -> tuple[bool, str]:
    """Attempt one candidate, logging useful but non-sensitive response diagnostics."""
    tmp = destination.with_suffix(destination.suffix + ".part")
    tmp.unlink(missing_ok=True)
    response = None
    try:
        kwargs = {"timeout": timeout, "allow_redirects": True, "stream": True}
        if candidate.method == "POST":
            kwargs["data"] = candidate.data
        elif candidate.data:
            kwargs["params"] = candidate.data
        response = session.request(candidate.method, candidate.url, **kwargs)
        status = response.status_code
        final_url = response.url
        content_type = response.headers.get("content-type", "")
        disposition = response.headers.get("content-disposition", "")
        response.raise_for_status()
        with response:
            iterator = response.iter_content(chunk_size=1024 * 512)
            first = next((chunk for chunk in iterator if chunk), b"")
            if not first.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
                sample = first[:180].decode("utf-8", errors="replace").replace("\n", " ").replace("\r", " ")
                return False, (
                    f"{candidate.label} -> HTTP {status}; content-type={content_type!r}; "
                    f"URL final={final_url}; content-disposition={disposition!r}; "
                    f"primeros bytes={first[:12]!r}; muestra={sample!r}"
                )
            size = 0
            with tmp.open("wb") as output:
                output.write(first)
                size += len(first)
                for chunk in iterator:
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > MAX_ARCHIVE_BYTES:
                        raise RuntimeError("El archivo supera el límite de seguridad de 1,5 GB.")
                    output.write(chunk)
        with zipfile.ZipFile(tmp) as archive:
            corrupt = archive.testzip()
            if corrupt:
                raise zipfile.BadZipFile(f"Miembro ZIP corrupto: {corrupt}")
            if not archive.namelist():
                raise zipfile.BadZipFile("ZIP vacío")
        tmp.replace(destination)
        return True, f"Descarga ZIP válida: {candidate.label} ({size:,} bytes)."
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        status_text = f"HTTP {response.status_code}" if response is not None else "sin respuesta HTTP"
        final_url = getattr(response, "url", candidate.url) if response is not None else candidate.url
        return False, f"{candidate.label} -> {status_text}; URL final={final_url}; error={type(exc).__name__}: {exc}"
    finally:
        if response is not None:
            response.close()


def download_sipsa_archives(
    start_year: int,
    end_year: int,
    cache_dir: str | Path,
    force: bool = False,
    progress=None,
) -> list[Path]:
    """Download the official SIPSA-P ZIP archives from DANE's public catalog."""
    start_year, end_year = int(start_year), int(end_year)
    if start_year > end_year:
        raise ValueError("El año inicial no puede ser mayor que el final.")
    if start_year < 2013 or end_year > 2024:
        raise ValueError("La ficha SIPSA-P consultada presenta archivos 2013–2024.")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(archive_name_for_year(y) for y in range(start_year, end_year + 1)))

    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/zip,application/octet-stream,*/*",
        "Referer": DANE_CATALOG_URL,
    })
    try:
        response = session.get(DANE_CATALOG_URL, timeout=45)
        response.raise_for_status()
        if "text/html" not in response.headers.get("content-type", "").lower():
            raise RuntimeError("El catálogo DANE no devolvió HTML.")
        html = response.text
    except Exception as exc:
        raise RuntimeError(f"No fue posible acceder al catálogo oficial del DANE: {exc}") from exc

    all_diagnostics: list[str] = []
    results: list[Path] = []
    for index, filename in enumerate(names, start=1):
        target = cache_dir / filename
        if target.exists() and target.stat().st_size > 1000 and not force:
            try:
                with zipfile.ZipFile(target) as archive:
                    if archive.testzip() is None and archive.namelist():
                        results.append(target)
                        if progress:
                            progress(index / len(names), desc=f"Usando caché verificada: {filename}")
                        continue
            except Exception:
                pass
            target.unlink(missing_ok=True)

        # Submit matching catalogue forms first; then try direct routes.
        candidates = _discover_form_candidates(html, filename, DANE_CATALOG_URL)
        candidates.extend(_discover_get_candidates(html, filename, DANE_CATALOG_URL))
        found = False
        for position, candidate in enumerate(candidates, start=1):
            if progress:
                progress(
                    (index - 1 + position / max(len(candidates), 1)) / len(names),
                    desc=f"Buscando descarga oficial: {filename} ({position}/{len(candidates)})",
                )
            ok, diagnostic = _request_and_validate(session, candidate, target)
            all_diagnostics.append(f"[{filename}] {diagnostic}")
            print(f"[SIPSA downloader] {diagnostic}")
            if ok:
                found = True
                break
            time.sleep(0.1)
        if not found:
            recent = "\n".join(all_diagnostics[-min(len(candidates), 12):])
            raise RuntimeError(
                f"No fue posible descargar un ZIP verificable para {filename} desde el catálogo oficial. "
                f"URL de catálogo: {DANE_CATALOG_URL}. Se probaron {len(candidates)} candidatos. "
                "Resumen técnico de las respuestas:\n" + (recent or "No se detectaron rutas candidatas.") +
                "\nEl proceso se detiene y no utiliza datos sintéticos."
            )
        results.append(target)
    return results
