"""
Web Crawler para AWS Bedrock Knowledge Base
============================================
Genera dos capas de datos:
  1. HTML crudo   → s3://<bucket>/raw-html/<domain>/<slug>.html
  2. JSONL limpio → s3://<bucket>/processed/<domain>/<slug>.jsonl
     (formato óptimo para Amazon Bedrock Knowledge Base / OpenSearch Ingestion)

Dependencias:
    pip install requests beautifulsoup4 lxml boto3 tqdm

Uso básico:
    from web_crawler_aws import WebCrawler

    # Desde sitemap
    crawler = WebCrawler(bucket="mi-bucket", prefix="mi-sitio")
    crawler.run_from_sitemap("https://ejemplo.com/sitemap.xml")

    # Desde lista de URLs
    urls = ["https://ejemplo.com/pagina1", "https://ejemplo.com/pagina2"]
    crawler.run_from_url_list(urls)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Iterator, Optional
from urllib.parse import urlparse, urljoin

import boto3
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Configuración de logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("web_crawler_aws")


# ──────────────────────────────────────────────────────────────────────────────
# Modelos de datos
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CrawlResult:
    """Resultado de crawlear una única URL."""

    url: str
    domain: str
    slug: str                        # identificador URL-safe para nombres de archivo
    status_code: int
    crawled_at: str                  # ISO-8601 UTC
    html_raw: str                    # HTML completo sin procesar
    title: str = ""
    description: str = ""
    text_chunks: list[str] = field(default_factory=list)  # texto limpio para embeddings
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None

    # ── Helpers ───────────────────────────────────────────────────────────────
    def s3_raw_key(self, prefix: str) -> str:
        """Clave S3 para el HTML crudo."""
        return f"{prefix}/raw-html/{self.domain}/{self.slug}.html"

    def s3_processed_key(self, prefix: str) -> str:
        """Clave S3 para el JSONL procesado (un registro por chunk)."""
        return f"{prefix}/processed/{self.domain}/{self.slug}.jsonl"

    def to_bedrock_records(self) -> list[dict]:
        """
        Convierte los chunks a registros compatibles con Amazon Bedrock
        Knowledge Base (formato recomendado para S3 data source + OpenSearch).

        Cada registro sigue el esquema:
        {
            "id": "<url>#<chunk_index>",
            "text": "<texto limpio>",
            "metadata": { ... }
        }
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
                    "metadata": {**base_meta, "chunk_index": i, "total_chunks": len(self.text_chunks)},
                }
            )
        return records


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────
_NAMESPACES = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "xhtml": "http://www.w3.org/1999/xhtml",
    "news": "http://www.google.com/schemas/sitemap-news/0.9",
    "image": "http://www.google.com/schemas/sitemap-image/1.1",
    "video": "http://www.google.com/schemas/sitemap-video/1.1",
}

_SLUG_RE = re.compile(r"[^\w\-]")
_MULTI_DASH = re.compile(r"-{2,}")


def url_to_slug(url: str, max_length: int = 120) -> str:
    """Convierte una URL a un slug seguro para nombres de archivo S3."""
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "index"
    slug = _SLUG_RE.sub("-", path).lower()
    slug = _MULTI_DASH.sub("-", slug).strip("-")
    # Añade hash corto para evitar colisiones
    short_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{slug[:max_length]}-{short_hash}"


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Divide el texto en chunks con overlap.
    chunk_size y overlap están en palabras (tokens aproximados).
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
    """Extrae el texto visible limpio de un BeautifulSoup."""
    # Elimina etiquetas no deseadas
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "noscript", "iframe", "svg", "form"]):
        tag.decompose()

    # Prioriza el contenido principal si existe
    main = soup.find("main") or soup.find("article") or soup.find(id=re.compile(r"content|main", re.I))
    target = main if main else soup.body or soup

    text = target.get_text(separator=" ", strip=True)
    # Normaliza espacios en blanco
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Crawler principal
# ──────────────────────────────────────────────────────────────────────────────
class WebCrawler:
    """
    Crawler asíncrono-secuencial con destino S3.

    Parameters
    ----------
    bucket      : Nombre del bucket S3 destino.
    prefix      : Prefijo (carpeta raíz) dentro del bucket.
    chunk_size  : Tamaño de chunk en palabras para los embeddings.
    overlap     : Solapamiento entre chunks en palabras.
    delay       : Segundos de espera entre peticiones (cortesía al servidor).
    timeout     : Timeout HTTP en segundos.
    max_retries : Reintentos en caso de error transitorio.
    user_agent  : User-Agent para las peticiones HTTP.
    aws_region  : Región AWS del bucket S3.
    session     : Sesión boto3 preconfigurada (opcional).
    dry_run     : Si True, no sube nada a S3 (útil para pruebas).
    """

    DEFAULT_UA = (
        "Mozilla/5.0 (compatible; AWSKnowledgeBaseCrawler/1.0; "
        "+https://aws.amazon.com/bedrock/)"
    )

    def __init__(
        self,
        bucket: str,
        prefix: str = "crawl",
        chunk_size: int = 512,
        overlap: int = 64,
        delay: float = 1.0,
        timeout: int = 20,
        max_retries: int = 3,
        user_agent: str = DEFAULT_UA,
        aws_region: str = "us-east-1",
        session: Optional[boto3.Session] = None,
        dry_run: bool = False,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.dry_run = dry_run

        # HTTP session con retry automático
        self._http = requests.Session()
        self._http.headers.update({"User-Agent": user_agent, "Accept-Language": "es,en;q=0.9"})

        # S3 client
        _session = session or boto3.Session(region_name=aws_region)
        self._s3 = _session.client("s3")

        # Estadísticas
        self._stats: dict = {"ok": 0, "error": 0, "skipped": 0, "bytes_raw": 0, "bytes_processed": 0}

    # ── Parsing de Sitemap ────────────────────────────────────────────────────
    def _fetch_sitemap_urls(self, sitemap_url: str) -> Iterator[str]:
        """
        Descarga y parsea un sitemap XML (incluyendo sitemap index).
        Soporta sitemaps anidados de forma recursiva.
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

        # Sitemap index → recursión
        tag = root.tag.lower()
        if "sitemapindex" in tag:
            for sitemap_el in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap"):
                loc = sitemap_el.findtext("{http://www.sitemaps.org/schemas/sitemap/0.9}loc", "").strip()
                if loc:
                    yield from self._fetch_sitemap_urls(loc)
        else:
            # Sitemap normal
            for url_el in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
                loc = url_el.findtext("{http://www.sitemaps.org/schemas/sitemap/0.9}loc", "").strip()
                if loc:
                    yield loc

    # ── Fetch y parseo de una página ─────────────────────────────────────────
    def _fetch_page(self, url: str) -> CrawlResult:
        """Descarga y parsea una URL con reintentos."""
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        slug = url_to_slug(url)
        crawled_at = datetime.now(timezone.utc).isoformat()

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._http.get(url, timeout=self.timeout, allow_redirects=True)
                break
            except requests.RequestException as exc:
                logger.warning("Intento %d/%d fallido para %s: %s", attempt, self.max_retries, url, exc)
                if attempt == self.max_retries:
                    return CrawlResult(
                        url=url, domain=domain, slug=slug,
                        status_code=0, crawled_at=crawled_at,
                        html_raw="", error=str(exc),
                    )
                time.sleep(self.delay * attempt)

        # Sólo procesamos HTML
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

        # Metadatos básicos
        title = (soup.title.string.strip() if soup.title and soup.title.string else "")
        description = ""
        meta_desc = soup.find("meta", attrs={"name": re.compile(r"description", re.I)})
        if meta_desc and meta_desc.get("content"):
            description = meta_desc["content"].strip()

        # Open Graph / Twitter Card extras
        og_meta = {}
        for m in soup.find_all("meta", property=re.compile(r"^og:")):
            key = m.get("property", "")[3:]  # strip 'og:'
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

    # ── Upload a S3 ───────────────────────────────────────────────────────────
    def _upload_to_s3(self, result: CrawlResult) -> None:
        """Sube el HTML crudo y el JSONL procesado a S3."""
        if self.dry_run:
            logger.debug("[DRY RUN] Saltando upload para %s", result.url)
            return

        # 1. HTML crudo
        raw_key = result.s3_raw_key(self.prefix)
        raw_bytes = result.html_raw.encode("utf-8")
        self._s3.put_object(
            Bucket=self.bucket,
            Key=raw_key,
            Body=raw_bytes,
            ContentType="text/html; charset=utf-8",
            Metadata={
                "source-url": result.url,
                "crawled-at": result.crawled_at,
                "status-code": str(result.status_code),
            },
        )
        self._stats["bytes_raw"] += len(raw_bytes)
        logger.debug("HTML subido → s3://%s/%s", self.bucket, raw_key)

        # 2. JSONL procesado (un JSON por línea, un registro por chunk)
        processed_key = result.s3_processed_key(self.prefix)
        records = result.to_bedrock_records()
        jsonl_body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records).encode("utf-8")
        self._s3.put_object(
            Bucket=self.bucket,
            Key=processed_key,
            Body=jsonl_body,
            ContentType="application/x-ndjson",
            Metadata={
                "source-url": result.url,
                "crawled-at": result.crawled_at,
                "total-chunks": str(len(records)),
            },
        )
        self._stats["bytes_processed"] += len(jsonl_body)
        logger.debug("JSONL subido → s3://%s/%s", self.bucket, processed_key)

    # ── Núcleo del crawler ────────────────────────────────────────────────────
    def _process_urls(self, urls: list[str]) -> list[CrawlResult]:
        """Itera sobre la lista de URLs, crawlea y sube a S3."""
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
                    self._upload_to_s3(result)
                    self._stats["ok"] += 1
                except Exception as exc:
                    logger.error("Error al subir %s a S3: %s", url, exc)
                    self._stats["error"] += 1

            results.append(result)
            time.sleep(self.delay)

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
        url_filter: Optional[callable] = None,
    ) -> list[CrawlResult]:
        """
        Crawlea todas las URLs encontradas en el sitemap (y sub-sitemaps).

        Parameters
        ----------
        sitemap_url : URL del sitemap XML principal.
        url_filter  : Función opcional ``f(url: str) -> bool`` para filtrar URLs.
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

    def run_from_file(self, filepath: str) -> list[CrawlResult]:
        """
        Carga URLs desde un archivo de texto (una URL por línea)
        y las crawlea.

        Parameters
        ----------
        filepath : Ruta al archivo .txt con las URLs.
        """
        with open(filepath, encoding="utf-8") as fh:
            urls = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
        return self._process_urls(urls)

    @property
    def stats(self) -> dict:
        """Retorna las estadísticas del último crawl."""
        return dict(self._stats)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de AWS (conveniencia)
# ──────────────────────────────────────────────────────────────────────────────
def generate_manifest(bucket: str, prefix: str, aws_region: str = "us-east-1") -> dict:
    """
    Genera un manifiesto S3 con todos los objetos procesados.
    Útil para auditoría o para alimentar Bedrock Batch Ingestion.
    """
    s3 = boto3.client("s3", region_name=aws_region)
    paginator = s3.get_paginator("list_objects_v2")
    processed_prefix = f"{prefix}/processed/"
    manifest = {"bucket": bucket, "prefix": processed_prefix, "files": []}

    for page in paginator.paginate(Bucket=bucket, Prefix=processed_prefix):
        for obj in page.get("Contents", []):
            manifest["files"].append(
                {"key": obj["Key"], "size": obj["Size"], "last_modified": obj["LastModified"].isoformat()}
            )

    return manifest


def sync_to_bedrock_knowledge_base(
    knowledge_base_id: str,
    data_source_id: str,
    aws_region: str = "us-east-1",
) -> str:
    """
    Dispara la ingesta de datos en Amazon Bedrock Knowledge Base.
    Requiere que el bucket S3 ya esté configurado como data source.

    Returns el ingestionJobId.
    """
    client = boto3.client("bedrock-agent", region_name=aws_region)
    response = client.start_ingestion_job(
        knowledgeBaseId=knowledge_base_id,
        dataSourceId=data_source_id,
    )
    job_id = response["ingestionJob"]["ingestionJobId"]
    logger.info("Ingesta iniciada → Job ID: %s", job_id)
    return job_id


# ──────────────────────────────────────────────────────────────────────────────
# Ejemplo de uso (ejecutar directamente)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Web Crawler → S3 (AWS Bedrock Knowledge Base)")
    parser.add_argument("--bucket", required=True, help="Bucket S3 destino")
    parser.add_argument("--prefix", default="crawl", help="Prefijo S3 (default: crawl)")
    parser.add_argument("--sitemap", help="URL del sitemap XML")
    parser.add_argument("--urls-file", help="Archivo .txt con URLs (una por línea)")
    parser.add_argument("--region", default="us-east-1", help="Región AWS")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay entre peticiones (s)")
    parser.add_argument("--chunk-size", type=int, default=512, help="Tamaño de chunk en palabras")
    parser.add_argument("--dry-run", action="store_true", help="No sube nada a S3")
    parser.add_argument("--kb-id", help="Knowledge Base ID de Bedrock (para auto-ingest)")
    parser.add_argument("--ds-id", help="Data Source ID de Bedrock (para auto-ingest)")
    args = parser.parse_args()

    if not args.sitemap and not args.urls_file:
        parser.error("Debes indicar --sitemap o --urls-file")

    crawler = WebCrawler(
        bucket=args.bucket,
        prefix=args.prefix,
        delay=args.delay,
        chunk_size=args.chunk_size,
        aws_region=args.region,
        dry_run=args.dry_run,
    )

    if args.sitemap:
        crawler.run_from_sitemap(args.sitemap)
    elif args.urls_file:
        crawler.run_from_file(args.urls_file)

    if args.kb_id and args.ds_id and not args.dry_run:
        sync_to_bedrock_knowledge_base(args.kb_id, args.ds_id, args.region)