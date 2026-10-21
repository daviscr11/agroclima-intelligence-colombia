# -*- coding: utf-8 -*-
"""Descarga automática y validada de microdatos SIPSA-P desde el catálogo oficial DANE.

No genera datos sintéticos. Descubre enlaces desde la ficha pública y valida ZIP mediante
firma y lectura de la tabla central. Si el portal cambia de estructura, falla de forma explícita.
"""
from __future__ import annotations

import re
import time
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

DANE_CATALOG_URL = "https://microdatos.dane.gov.co/index.php/catalog/776/get-microdata"
DANE_FILE_PREFIX = "BaseDatos-SIPSA_P-Mensual-"
USER_AGENT = "AgroClima-SIPSA research app/1.0 (automated public data retrieval)"


def archive_name_for_year(year: int) -> str:
    year = int(year)
    if 2013 <= year <= 2017:
        return f"{DANE_FILE_PREFIX}2013_2017.zip"
    if 2018 <= year <= 2024:
        return f"{DANE_FILE_PREFIX}{year}.zip"
    raise ValueError(f"El catálogo SIPSA-P consultado contiene los años 2013–2024; año fuera de rango: {year}.")


def _normal_url(candidate: str, base_url: str) -> str | None:
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
    if parsed.netloc and not (parsed.netloc.endswith("dane.gov.co") or parsed.netloc.endswith("microdatos.dane.gov.co")):
        # Avoid following arbitrary third-party links from a catalog page.
        return None
    return candidate


def _discover_catalog_file_id(html: str, target_filename: str) -> str | None:
    """Find the NADA data-file ID (e.g. F3) associated with the exact SIPSA filename.

    The catalogue's download button is rendered through UI controls, not always as a
    direct ZIP hyperlink. Its data-dictionary URL, however, exposes the stable file ID.
    We match the filename in the link/query or its small surrounding HTML block.
    """
    soup = BeautifulSoup(html, "html.parser")
    stem = Path(target_filename).stem
    if stem.lower().startswith("basedatos-"):
        stem = stem[len("BaseDatos-"):]
    target = stem.lower()
    target_compact = re.sub(r"[^a-z0-9]", "", target)

    for anchor in soup.find_all("a", href=True):
        href = unquote(str(anchor.get("href", "")))
        match = re.search(
            r"/catalog/\d+/(?:data-dictionary|data_dictionary|datafile|data-file)/([A-Za-z0-9_-]+)",
            href,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        query = parse_qs(urlparse(href).query)
        values = [href, anchor.get_text(" ", strip=True)]
        values.extend(query.get("file_name", []))
        # Keep the contextual search local; a full page contains all years and could
        # otherwise associate the wrong file ID with the requested year.
        parent = anchor.parent
        if parent is not None:
            values.append(parent.get_text(" ", strip=True)[:1200])
            if parent.parent is not None:
                values.append(parent.parent.get_text(" ", strip=True)[:1800])
        context = " ".join(values).lower()
        context_compact = re.sub(r"[^a-z0-9]", "", context)
        if target in context or target_compact in context_compact:
            return match.group(1)
    return None


def _discover_candidate_urls(html: str, target_filename: str, base_url: str) -> list[str]:
    """Discover direct file URLs, including links embedded in a file's surrounding HTML block."""
    soup = BeautifulSoup(html, "html.parser")
    scored: dict[str, int] = {}
    target_norm = target_filename.lower()

    def add(raw_url: str | None, score: int) -> None:
        url = _normal_url(raw_url or "", base_url)
        if url:
            scored[url] = max(score, scored.get(url, -1))

    # The DANE catalogue is powered by NADA. Its data-dictionary link exposes a file ID
    # (for example F3 for SIPSA_P-Mensual-2019); use the official study download route
    # with that ID instead of guessing a ZIP URL from the displayed filename.
    file_id = _discover_catalog_file_id(html, target_filename)
    if file_id:
        for raw in [
            f"/index.php/catalog/776/download/{file_id}",
            f"/index.php/catalog/776/download/FILE/{file_id}",
        ]:
            add(raw, 1000)

    # 1) Links with ZIP, download, or filename in the URL.
    for tag in soup.find_all(True):
        attrs = " ".join(str(tag.attrs.get(k, "")) for k in ["href", "src", "onclick", "data-href", "data-url", "data-download-url"])
        txt = tag.get_text(" ", strip=True) if hasattr(tag, "get_text") else ""
        context = f"{attrs} {txt}".lower()
        for attr in ["href", "src", "data-href", "data-url", "data-download-url"]:
            val = tag.attrs.get(attr)
            if isinstance(val, str):
                val_low = val.lower()
                score = 0
                if target_norm in val_low:
                    score += 100
                # Do not queue unrelated ZIPs from another year in the same catalog.
                # Links without the filename are discovered only inside the exact file block below.
                if score:
                    add(val, score)

    # 2) Find the filename on the page and inspect nearest containers for related links/buttons.
    for node in soup.find_all(string=re.compile(re.escape(target_filename), re.I)):
        parent = node.parent
        for depth in range(8):
            if parent is None:
                break
            container_text = parent.get_text(" ", strip=True).lower()
            if target_norm in container_text:
                found_download_in_container = False
                for elem in parent.find_all(True):
                    element_text = elem.get_text(" ", strip=True).lower() if hasattr(elem, "get_text") else ""
                    for attr in ["href", "src", "data-href", "data-url", "data-download-url", "onclick"]:
                        val = elem.attrs.get(attr)
                        if not isinstance(val, str):
                            continue
                        # onclick may include a quoted relative URL; extract URL-like substrings.
                        extracted = re.findall(r"(?:https?://[^\s\"'<>]+|(?:/|\.\.?/)[^\s\"'<>]+)", val)
                        candidates = extracted if attr == "onclick" and extracted else [val]
                        for raw in candidates:
                            lower = raw.lower()
                            is_download_like = ("download" in lower or "descargar" in lower or ".zip" in lower
                                                or target_norm in lower or "descargar" in element_text or "download" in element_text)
                            if not is_download_like:
                                continue
                            score = 120 if target_norm in lower else 45
                            add(raw, score)
                            found_download_in_container = True
                # Stop at the nearest file card/row once it exposes a download-like URL.
                if found_download_in_container or parent.name in {"tr", "article", "section", "li"}:
                    break
            parent = parent.parent

    # 3) Do not scan a broad page fragment for arbitrary URLs: that can associate another
    # year's ZIP with the requested filename. Context-free fallback URLs are tried below
    # and accepted only if they return a valid ZIP.

    # 4) Version-tolerant guesses for common catalog download handlers. Each candidate is
    # validated as an actual ZIP before being accepted; a HTML page is never treated as data.
    escaped = quote(target_filename)
    guesses = [
        f"/index.php/catalog/776/download/FILE?file_name={escaped}",
        f"/index.php/catalog/776/download?file_name={escaped}",
        f"/index.php/catalog/776/download/FILE/{escaped}",
        f"/index.php/catalog/776/download/{escaped}",
        f"/index.php/catalog/776/get-microdata?file_name={escaped}",
    ]
    for raw in guesses:
        add(raw, 1)

    return [url for url, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)]


def _download_and_validate(session: requests.Session, url: str, destination: Path, timeout: int = 120) -> bool:
    """Download to a temporary path and accept only a valid ZIP archive."""
    tmp = destination.with_suffix(destination.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=timeout, allow_redirects=True) as response:
            response.raise_for_status()
            first = b""
            with tmp.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 512):
                    if not chunk:
                        continue
                    if not first:
                        first = chunk[:8]
                    f.write(chunk)
                    if f.tell() > 1_500_000_000:
                        raise RuntimeError("El archivo supera el límite de seguridad de 1,5 GB.")
            if not first.startswith(b"PK\x03\x04") and not first.startswith(b"PK\x05\x06") and not first.startswith(b"PK\x07\x08"):
                tmp.unlink(missing_ok=True)
                return False
            with zipfile.ZipFile(tmp) as zf:
                bad_member = zf.testzip()
                if bad_member:
                    raise zipfile.BadZipFile(f"Miembro ZIP corrupto: {bad_member}")
                if not zf.namelist():
                    raise zipfile.BadZipFile("ZIP vacío")
            tmp.replace(destination)
            return True
    except Exception:
        tmp.unlink(missing_ok=True)
        return False


def download_sipsa_archives(start_year: int, end_year: int, cache_dir: str | Path, force: bool = False, progress=None) -> list[Path]:
    """Download SIPSA-P ZIP files corresponding to selected years from DANE's public catalog.

    The 2013–2017 archive is shared across five years. Subsequent archives are annual through 2024.
    Existing verified ZIPs are reused unless force=True. Never creates synthetic observations.
    """
    start_year, end_year = int(start_year), int(end_year)
    if start_year > end_year:
        raise ValueError("El año inicial no puede ser mayor que el año final.")
    if start_year < 2013 or end_year > 2024:
        raise ValueError("La ficha SIPSA-P consultada presenta archivos 2013–2024. Seleccione un intervalo dentro de ese rango.")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(archive_name_for_year(y) for y in range(start_year, end_year + 1)))
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/zip,application/octet-stream,*/*"})
    try:
        response = session.get(DANE_CATALOG_URL, timeout=45)
        response.raise_for_status()
        if "text/html" not in response.headers.get("content-type", "").lower():
            raise RuntimeError("El catálogo DANE no devolvió una página HTML legible.")
        html = response.text
    except Exception as exc:
        raise RuntimeError(f"No fue posible acceder al catálogo oficial SIPSA-P del DANE ({DANE_CATALOG_URL}): {exc}") from exc

    results = []
    for idx, filename in enumerate(names, 1):
        target = cache_dir / filename
        if target.exists() and target.stat().st_size > 1000 and not force:
            try:
                with zipfile.ZipFile(target) as zf:
                    if zf.testzip() is None and zf.namelist():
                        results.append(target)
                        if progress:
                            progress(idx / len(names), desc=f"Usando caché verificada: {filename}")
                        continue
            except Exception:
                pass
            target.unlink(missing_ok=True)

        candidates = _discover_candidate_urls(html, filename, DANE_CATALOG_URL)
        found = False
        for j, url in enumerate(candidates, 1):
            if progress:
                progress((idx - 1 + j / max(len(candidates), 1)) / len(names), desc=f"Buscando descarga oficial: {filename}")
            if _download_and_validate(session, url, target):
                found = True
                break
            time.sleep(0.15)
        if not found:
            raise RuntimeError(
                f"No fue posible encontrar una descarga ZIP verificable para {filename}. "
                f"El catálogo oficial está disponible en {DANE_CATALOG_URL}. "
                "Se detectaron rutas candidatas del catálogo, pero ninguna devolvió un ZIP válido. "
                "Revise el registro de descarga y el mecanismo actual de NADA; la aplicación no usará datos sintéticos."
            )
        results.append(target)
    return results
