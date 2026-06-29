"""
Web Crawler → Almacenamiento Local
====================================
Genera dos capas de datos en disco:
  1. HTML crudo     → <output_dir>/raw-html/<domain>/<slug>.html
  2. JSONL procesado → <output_dir>/processed/<domain>/<slug>.jsonl
     (formato listo para cargar en una base de datos vectorial)

Dependencias:
    pip install requests beautifulsoup4 lxml tqdm

Uso básico:
    from web_crawler_local import WebCrawler

    # Desde sitemap
    crawler = WebCrawler(output_dir="./crawl_output")
    crawler.run_from_sitemap("https://ejemplo.com/sitemap.xml")

    # Desde lista de URLs
    urls = ["https://ejemplo.com/pagina1", "https://ejemplo.com/pagina2"]
    crawler.run_from_url_list(urls)

    # Desde archivo .txt (una URL por línea)
    crawler.run_from_file("mis_urls.txt")

Estructura de salida:
    crawl_output/
    ├── raw-html/
    │   └── ejemplo.com/
    │       ├── pagina1-a1b2c3d4.html
    │       └── pagina2-e5f6g7h8.html
    ├── processed/
    │   └── ejemplo.com/
    │       ├── pagina1-a1b2c3d4.jsonl
    │       └── pagina2-e5f6g7h8.jsonl
    └── manifest.json          ← índice de todo lo crawleado
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("web_crawler_local")


# ──────────────────────────────────────────────────────────────────────────────
# Modelos de datos
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CrawlResult:
    """Resultado de crawlear una única URL."""

    url: str
    domain: str
    slug: str                          # identificador URL-safe para nombres de archivo
    status_code: int
    crawled_at: str                    # ISO-8601 UTC
    html_raw: str                      # HTML completo sin procesar
    title: str = ""
    description: str = ""
    text_chunks: list[str] = field(default_factory=list)  # texto limpio para embeddings
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None

    # ── Rutas locales ─────────────────────────────────────────────────────────
    def local_raw_path(self, output_dir: Path) -> Path:
        """Ruta local para el HTML crudo."""
        return output_dir / "raw-html" / self.domain / f"{self.slug}.html"

    def local_processed_path(self, output_dir: Path) -> Path:
        """Ruta local para el JSONL procesado."""
        return output_dir / "processed" / self.domain / f"{self.slug}.jsonl"

    def to_vector_records(self) -> list[dict]:
        """
        Convierte los chunks a registros listos para una base de datos vectorial.

        Formato JSONL (un objeto JSON por línea):
        {
            "id":       "<url>#<chunk_index>",
            "text":     "<texto limpio del chunk>",
            "metadata": { "source_url", "title", "crawled_at", ... }
        }
        Compatible con: Chroma, Qdrant, Weaviate, Pinecone, pgvector,
                        Amazon Bedrock Knowledge Base, OpenSearch, etc.
        """
        base_meta = {
            "source_url": self.url,
            "title": self.title,
            "description": self.description,
            "crawled_at": self.crawled_at,
            **self.metadata,
        }
        records = []
        for i, chunk in enumerate(self.text_chunks):
            records.append(
                {
                    "id": f"{self.url}#{i}",
                    "text": chunk,
                    "metadata": {
                        **base_meta,
                        "chunk_index": i,
                        "total_chunks": len(self.text_chunks),
                    },
                }
            )
        return records

    def to_manifest_entry(self, output_dir: Path) -> dict:
        """Entrada resumida para el manifiesto global."""
        return {
            "url": self.url,
            "slug": self.slug,
            "domain": self.domain,
            "status_code": self.status_code,
            "crawled_at": self.crawled_at,
            "title": self.title,
            "total_chunks": len(self.text_chunks),
            "error": self.error,
            "raw_path": str(self.local_raw_path(output_dir)),
            "processed_path": str(self.local_processed_path(output_dir)),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────
_SLUG_RE = re.compile(r"[^\w\-]")
_MULTI_DASH = re.compile(r"-{2,}")


def url_to_slug(url: str, max_length: int = 120) -> str:
    """Convierte una URL a un slug seguro para nombres de archivo."""
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "index"
    slug = _SLUG_RE.sub("-", path).lower()
    slug = _MULTI_DASH.sub("-", slug).strip("-")
    short_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{slug[:max_length]}-{short_hash}"


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Divide texto en chunks con overlap.
    chunk_size y overlap están en palabras (≈ tokens).
    """
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def extract_text_from_soup(soup: BeautifulSoup) -> str:
    """Extrae el texto visible y limpio de un objeto BeautifulSoup."""
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "noscript", "iframe", "svg", "form"]):
        tag.decompose()

    # Prioriza contenido principal si existe
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"content|main", re.I))
    )
    target = main if main else (soup.body or soup)

    text = target.get_text(separator=" ", strip=True)
    return re.sub(r"\s{2,}", " ", text).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Crawler principal
# ──────────────────────────────────────────────────────────────────────────────
class WebCrawler:
    """
    Crawler secuencial con almacenamiento local en disco.

    Parameters
    ----------
    output_dir  : Carpeta raíz donde se guardarán los archivos.
                  Se crea automáticamente si no existe.
    chunk_size  : Tamaño de chunk en palabras para los embeddings (default 512).
    overlap     : Solapamiento entre chunks en palabras (default 64).
    delay       : Segundos de espera entre peticiones (default 1.0).
    timeout     : Timeout HTTP en segundos (default 20).
    max_retries : Reintentos en caso de error transitorio (default 3).
    user_agent  : User-Agent para las peticiones HTTP.
    """

    DEFAULT_UA = (
        "Mozilla/5.0 (compatible; LocalVectorCrawler/1.0; "
        "https://github.com/tu-proyecto)"
    )

    def __init__(
        self,
        output_dir: str | Path = "./crawl_output",
        chunk_size: int = 512,
        overlap: int = 64,
        delay: float = 1.0,
        timeout: int = 20,
        max_retries: int = 3,
        user_agent: str = DEFAULT_UA,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries

        # Crear estructura de carpetas
        (self.output_dir / "raw-html").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "processed").mkdir(parents=True, exist_ok=True)
        logger.info("Directorio de salida: %s", self.output_dir.resolve())

        # HTTP session
        self._http = requests.Session()
        self._http.headers.update({
            "User-Agent": user_agent,
            "Accept-Language": "es,en;q=0.9",
        })

        # Estadísticas
        self._stats: dict = {
            "ok": 0, "error": 0, "skipped": 0,
            "bytes_raw": 0, "bytes_processed": 0,
        }

        # Registro de entradas para el manifiesto
        self._manifest_entries: list[dict] = []

    # ── Parsing de Sitemap ────────────────────────────────────────────────────
    def _fetch_sitemap_urls(self, sitemap_url: str) -> Iterator[str]:
        """
        Descarga y parsea un sitemap XML.
        Soporta sitemap index (recursivo) y sitemaps normales.
        """
        logger.info("Descargando sitemap: %s", sitemap_url)
        try:
            resp = self._http.get(sitemap_url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Error al descargar sitemap %s: %s", sitemap_url, exc)
            return

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as exc:
            logger.error("XML inválido en %s: %s", sitemap_url, exc)
            return

        ns = "http://www.sitemaps.org/schemas/sitemap/0.9"
        tag = root.tag.lower()

        if "sitemapindex" in tag:
            # Sitemap index → recursión sobre cada sub-sitemap
            for sitemap_el in root.iter(f"{{{ns}}}sitemap"):
                loc = sitemap_el.findtext(f"{{{ns}}}loc", "").strip()
                if loc:
                    yield from self._fetch_sitemap_urls(loc)
        else:
            # Sitemap de URLs normal
            for url_el in root.iter(f"{{{ns}}}url"):
                loc = url_el.findtext(f"{{{ns}}}loc", "").strip()
                if loc:
                    yield loc

    # ── Fetch y parseo de página ──────────────────────────────────────────────
    def _fetch_page(self, url: str) -> CrawlResult:
        """Descarga, parsea y estructura el contenido de una URL."""
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        slug = url_to_slug(url)
        crawled_at = datetime.now(timezone.utc).isoformat()

        # Reintentos con backoff lineal
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._http.get(url, timeout=self.timeout, allow_redirects=True)
                break
            except requests.RequestException as exc:
                logger.warning("Intento %d/%d fallido para %s: %s",
                               attempt, self.max_retries, url, exc)
                if attempt == self.max_retries:
                    return CrawlResult(
                        url=url, domain=domain, slug=slug,
                        status_code=0, crawled_at=crawled_at,
                        html_raw="", error=str(exc),
                    )
                time.sleep(self.delay * attempt)

        # Solo procesamos HTML
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return CrawlResult(
                url=url, domain=domain, slug=slug,
                status_code=resp.status_code, crawled_at=crawled_at,
                html_raw="",
                error=f"Content-Type no soportado: {content_type}",
            )

        html_raw = resp.text
        soup = BeautifulSoup(html_raw, "lxml")

        # Título
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Meta description
        description = ""
        meta_desc = soup.find("meta", attrs={"name": re.compile(r"description", re.I)})
        if meta_desc and meta_desc.get("content"):
            description = meta_desc["content"].strip()

        # Open Graph extras
        og_meta = {}
        for m in soup.find_all("meta", property=re.compile(r"^og:")):
            key = m.get("property", "")[3:]
            og_meta[f"og_{key}"] = m.get("content", "")

        # Texto limpio y chunking
        clean_text = extract_text_from_soup(soup)
        text_chunks = chunk_text(clean_text, self.chunk_size, self.overlap)

        return CrawlResult(
            url=url,
            domain=domain,
            slug=slug,
            status_code=resp.status_code,
            crawled_at=crawled_at,
            html_raw=html_raw,
            title=title,
            description=description,
            text_chunks=text_chunks,
            metadata={
                "final_url": resp.url,
                "content_length": len(html_raw),
                **og_meta,
            },
        )

    # ── Guardado local ────────────────────────────────────────────────────────
    def _save_to_disk(self, result: CrawlResult) -> None:
        """
        Persiste en disco:
          - HTML crudo  → raw-html/<domain>/<slug>.html
          - JSONL       → processed/<domain>/<slug>.jsonl
        """
        # ── 1. HTML crudo ─────────────────────────────────────────────────────
        raw_path = result.local_raw_path(self.output_dir)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_bytes = result.html_raw.encode("utf-8")
        raw_path.write_bytes(raw_bytes)
        self._stats["bytes_raw"] += len(raw_bytes)
        logger.debug("HTML guardado → %s", raw_path)

        # ── 2. JSONL procesado ────────────────────────────────────────────────
        processed_path = result.local_processed_path(self.output_dir)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        records = result.to_vector_records()
        jsonl_body = "\n".join(
            json.dumps(r, ensure_ascii=False) for r in records
        ).encode("utf-8")
        processed_path.write_bytes(jsonl_body)
        self._stats["bytes_processed"] += len(jsonl_body)
        logger.debug("JSONL guardado → %s", processed_path)

    # ── Manifiesto ────────────────────────────────────────────────────────────
    def _write_manifest(self) -> Path:
        """
        Escribe (o actualiza) manifest.json en la raíz del output_dir.
        Contiene el índice completo de todo lo crawleado.
        """
        manifest_path = self.output_dir / "manifest.json"
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_entries": len(self._manifest_entries),
            "stats": self._stats,
            "entries": self._manifest_entries,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Manifiesto actualizado → %s", manifest_path)
        return manifest_path

    # ── Núcleo del crawler ────────────────────────────────────────────────────
    def _process_urls(self, urls: list[str]) -> list[CrawlResult]:
        """Itera sobre las URLs, crawlea y guarda en disco."""
        results: list[CrawlResult] = []
        unique_urls = list(dict.fromkeys(urls))  # deduplica preservando orden
        logger.info("Iniciando crawl de %d URLs únicas", len(unique_urls))

        for url in tqdm(unique_urls, desc="Crawling", unit="url"):
            result = self._fetch_page(url)

            if result.error:
                logger.warning("Error en %s: %s", url, result.error)
                self._stats["error"] += 1
            elif not result.text_chunks:
                logger.info("Sin contenido útil en %s (skipped)", url)
                self._stats["skipped"] += 1
            else:
                try:
                    self._save_to_disk(result)
                    self._stats["ok"] += 1
                except OSError as exc:
                    logger.error("Error al guardar %s: %s", url, exc)
                    self._stats["error"] += 1

            self._manifest_entries.append(result.to_manifest_entry(self.output_dir))
            results.append(result)
            time.sleep(self.delay)

        self._write_manifest()
        self._log_summary()
        return results

    def _log_summary(self) -> None:
        s = self._stats
        logger.info(
            "Crawl completado → OK: %d | Errores: %d | Sin contenido: %d | "
            "Raw: %.1f KB | Processed: %.1f KB",
            s["ok"], s["error"], s["skipped"],
            s["bytes_raw"] / 1024, s["bytes_processed"] / 1024,
        )

    # ── Puntos de entrada públicos ────────────────────────────────────────────
    def run_from_sitemap(
        self,
        sitemap_url: str,
        url_filter: Optional[Callable[[str], bool]] = None,
    ) -> list[CrawlResult]:
        """
        Crawlea todas las URLs del sitemap XML (incluye sub-sitemaps).

        Parameters
        ----------
        sitemap_url : URL del sitemap XML principal.
        url_filter  : Función opcional ``f(url) -> bool`` para filtrar URLs.
                      Ejemplo: lambda u: "/blog/" in u
        """
        urls = list(self._fetch_sitemap_urls(sitemap_url))
        if url_filter:
            urls = [u for u in urls if url_filter(u)]
        return self._process_urls(urls)

    def run_from_url_list(self, urls: list[str]) -> list[CrawlResult]:
        """
        Crawlea una lista de URLs definida manualmente.

        Parameters
        ----------
        urls : Lista de URLs absolutas a procesar.
        """
        return self._process_urls(urls)

    def run_from_file(self, filepath: str | Path) -> list[CrawlResult]:
        """
        Carga URLs desde un archivo de texto (una URL por línea,
        las líneas que empiezan con '#' se ignoran como comentarios).

        Parameters
        ----------
        filepath : Ruta al archivo .txt con las URLs.
        """
        with open(filepath, encoding="utf-8") as fh:
            urls = [
                line.strip()
                for line in fh
                if line.strip() and not line.startswith("#")
            ]
        return self._process_urls(urls)

    @property
    def stats(self) -> dict:
        """Estadísticas del último crawl."""
        return dict(self._stats)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: cargar el JSONL para usar con librerías de vectores
# ──────────────────────────────────────────────────────────────────────────────
def load_processed_records(output_dir: str | Path) -> list[dict]:
    """
    Lee todos los archivos JSONL del directorio 'processed' y los devuelve
    como una lista de registros lista para vectorizar.

    Ejemplo de uso con sentence-transformers + Chroma:

        from web_crawler_local import load_processed_records
        import chromadb
        from sentence_transformers import SentenceTransformer

        records = load_processed_records("./crawl_output")
        model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

        client = chromadb.Client()
        col = client.create_collection("mi_sitio")
        col.add(
            ids=[r["id"] for r in records],
            documents=[r["text"] for r in records],
            metadatas=[r["metadata"] for r in records],
            embeddings=model.encode([r["text"] for r in records]).tolist(),
        )
    """
    output_dir = Path(output_dir)
    records: list[dict] = []
    for jsonl_file in sorted((output_dir / "processed").rglob("*.jsonl")):
        with jsonl_file.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    logger.info("Registros cargados: %d (desde %s)", len(records), output_dir / "processed")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Web Crawler → Almacenamiento local (base de datos vectorial)"
    )
    parser.add_argument("--output-dir", default="./crawl_output",
                        help="Carpeta de salida (default: ./crawl_output)")
    parser.add_argument("--sitemap", help="URL del sitemap XML")
    parser.add_argument("--urls-file", help="Archivo .txt con URLs (una por línea)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay entre peticiones en segundos (default: 1.0)")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Tamaño de chunk en palabras (default: 512)")
    parser.add_argument("--overlap", type=int, default=64,
                        help="Solapamiento entre chunks en palabras (default: 64)")
    parser.add_argument("--timeout", type=int, default=20,
                        help="Timeout HTTP en segundos (default: 20)")
    args = parser.parse_args()

    if not args.sitemap and not args.urls_file:
        parser.error("Debes indicar --sitemap o --urls-file")

    crawler = WebCrawler(
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        delay=args.delay,
        timeout=args.timeout,
    )

    if args.sitemap:
        crawler.run_from_sitemap(args.sitemap)
    elif args.urls_file:
        crawler.run_from_file(args.urls_file)

    # Mostrar resumen de estadísticas
    print("\n── Estadísticas ──────────────────────────────")
    for k, v in crawler.stats.items():
        print(f"  {k:20s}: {v}")