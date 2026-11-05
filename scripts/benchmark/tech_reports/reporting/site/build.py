# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Build a local, self-contained HTML reading copy without changing evidence."""

from __future__ import annotations

import os

import argparse
import hashlib
import html
import json
import re
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import markdown
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown.extensions.toc import slugify_unicode

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("SIM_REPORT_ROOT", HERE.parents[4])).resolve()
STATUS = {"complete": "本轮完成", "partial": "部分完成", "pending": "待开展"}


class SiteBuilder:
    """Export the catalog and its local evidence links into one static root."""

    def __init__(self, output: Path) -> None:
        self.output = output.resolve()
        self.catalog = json.loads((HERE / "catalog.json").read_text())
        self.templates = Environment(
            loader=FileSystemLoader(HERE / "templates"),
            autoescape=select_autoescape(["html"]),
        )
        self.date = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
        self.documents: dict[Path, dict] = {}
        self.assets: set[Path] = set()
        self.directories: set[Path] = set()
        self.queue: deque[Path] = deque()
        self.unpublished: list[dict] = []
        retention_path = ROOT / "reports/media-retention.json"
        self.retention = (
            json.loads(retention_path.read_text()) if retention_path.exists() else {}
        )
        self.media_notices: dict[tuple[Path, str], str] = {}
        self.main_sources = {
            (ROOT / item["source"]).resolve(): item
            for item in self.catalog
            if item.get("source")
        }
        for item in self.catalog:
            item["url"] = f"/experiments/{item['id'].lower()}.html"
            item["status_label"] = STATUS[item["status"]]
        self.groups = [
            {"name": name, "items": [x for x in self.catalog if x["category"] == name]}
            for name in ["加载", "渲染质量", "物理", "Demo"]
        ]

    def register_document(self, path: Path, owner: dict) -> str:
        if path not in self.documents:
            relative = path.relative_to(ROOT).as_posix()
            slug = re.sub(r"[^a-zA-Z0-9-]+", "-", relative.removesuffix(".md"))
            url = (
                self.main_sources[path]["url"]
                if path in self.main_sources
                else f"/notes/{slug}.html"
            )
            self.documents[path] = {"url": url, "owner": owner, "source": relative}
            self.queue.append(path)
        return self.documents[path]["url"]

    def file_url(self, path: Path) -> str:
        return "/files/" + quote(path.relative_to(ROOT).as_posix(), safe="/")

    def local_url(self, reference: str, source: Path, owner: dict) -> str | None:
        parsed = urlsplit(reference)
        if parsed.scheme in {"https", "http", "mailto", "data"} or not parsed.path:
            return reference
        if parsed.scheme not in {"", "file"} or parsed.netloc:
            return None
        raw_path = re.sub(r":\d+(?::\d+)?$", "", unquote(parsed.path))
        path = (
            Path(raw_path) if raw_path.startswith("/") else source.parent / raw_path
        ).resolve()
        if not path.is_relative_to(ROOT):
            self.unpublished.append(
                {"source": str(source.relative_to(ROOT)), "reference": reference}
            )
            return None
        relative_name = path.relative_to(ROOT).as_posix()
        for rule in self.retention.get("retired_rules", []):
            if relative_name == rule.get("path") or (
                rule.get("directory")
                and path.is_relative_to(ROOT / rule["directory"])
                and path.suffix.lower() in rule["suffixes"]
            ):
                self.media_notices[source, reference] = "已清理：" + rule["reason"]
                return self.register_document(ROOT / "reports/media-cleanup.md", owner)
        for record in self.retention.get("redirects", []):
            if relative_name == record["path"]:
                path = (ROOT / record["replacement"]).resolve()
                self.media_notices[source, reference] = record["note"]
                break
        for record in self.retention.get("compressed_files", []):
            if relative_name == record["path"]:
                path = (ROOT / record["archive"]).resolve()
                self.media_notices[source, reference] = (
                    "gzip 无损压缩；解压后与原文件 SHA-256 一致"
                )
                break
        if not path.is_relative_to(ROOT):
            raise ValueError(f"Media replacement leaves project: {path}")
        if not path.exists():
            raise FileNotFoundError(
                f"Broken source link: {source.relative_to(ROOT)} -> {reference}"
            )
        relative = path.relative_to(ROOT)
        # Project handover instructions are not experiment publication material.
        if relative.as_posix() == "README.md":
            return "/"
        if relative.as_posix() == "PROGRESS.md":
            return "/#directory-heading"
        if (
            relative.as_posix() == "AGENTS.md"
            or relative.parts[0] == ".agents"
            or path.name.startswith("PROGRESS.history-")
        ):
            return None
        if path.is_dir():
            self.directories.add(path)
            url = self.file_url(path) + "/"
        elif path.suffix.lower() == ".md" and (
            relative.parts[0] == "reports"
            or (
                relative.parts[:2] == ("outputs", "benchmarks")
                and any(x in relative.parts for x in ("reports", "videos"))
            )
        ):
            url = self.register_document(path, owner)
        else:
            self.assets.add(path)
            url = self.file_url(path)
        if parsed.fragment and (path.suffix.lower() != ".md" or path in self.documents):
            url += "#" + quote(unquote(parsed.fragment), safe="-_.~")
        return url

    def render_document(self, source: Path) -> None:
        doc = self.documents[source]
        formulas: list[tuple[str, str]] = []

        def protect(match: re.Match) -> str:
            token = f"FORMULAPLACEHOLDER{len(formulas)}END"
            display = match.group(1) is not None
            tex = (match.group(1) if display else match.group(2)).strip()
            tag = "span"
            markup = f'<{tag} class="math-{"display" if display else "inline"}" data-tex="{html.escape(tex, quote=True)}">{html.escape(tex)}</{tag}>'
            formulas.append((token, markup))
            return token

        text = source.read_text(encoding="utf-8")
        parts = re.split(r"(```.*?```|`[^`\n]+`)", text, flags=re.S)
        for i in range(0, len(parts), 2):
            parts[i] = re.sub(
                r"(?<!\\)\$\$(.+?)\$\$|(?<!\\)\$(?!\s)([^$\n]+?)\$",
                protect,
                parts[i],
                flags=re.S,
            )
        md = markdown.Markdown(
            extensions=["tables", "fenced_code", "toc", "sane_lists", "footnotes"],
            extension_configs={"toc": {"slugify": slugify_unicode, "toc_depth": "2-3"}},
        )
        rendered = md.convert("".join(parts))
        for token, markup in formulas:
            rendered = rendered.replace(token, markup)
        soup = BeautifulSoup(rendered, "html.parser")
        heading = soup.find("h1")
        doc["title"] = heading.get_text(" ", strip=True) if heading else source.stem
        if heading:
            heading.decompose()
        for node in soup.select("a[href], img[src], video[src], source[src]"):
            attribute = "href" if node.name == "a" else "src"
            original = node[attribute]
            rewritten = self.local_url(original, source, doc["owner"])
            notice = self.media_notices.get((source, original))
            if notice and notice.startswith("已清理"):
                if node.name != "a":
                    replacement = soup.new_tag("a", href=rewritten)
                    replacement.string = node.get("alt", "历史媒体") + "（已清理）"
                    target = (
                        node.parent
                        if node.name == "source" and node.parent.name == "video"
                        else node
                    )
                    target.replace_with(replacement)
                    node = replacement
                else:
                    node.append("（已清理）")
                node["title"] = notice
                node["href"] = rewritten
                continue
            if notice:
                node["title"] = notice
                if node.name == "a" and "gzip" in notice:
                    node.append("（gzip 无损压缩）")
            if rewritten is None:
                if node.name != "a":
                    raise ValueError(f"Image/media outside export scope: {original}")
                node.name = "span"
                del node[attribute]
                node["class"] = "local-reference"
                node["title"] = "本机环境引用，未发布为下载文件"
            else:
                node[attribute] = rewritten
            if node.name == "a" and original.startswith(("http://", "https://")):
                node["target"] = "_blank"
                node["rel"] = "noopener noreferrer"
        for table in soup.find_all("table"):
            wrapper = soup.new_tag(
                "div",
                attrs={
                    "class": "table-scroll",
                    "tabindex": "0",
                    "role": "region",
                    "aria-label": "数据表，可横向滚动",
                },
            )
            table.wrap(wrapper)
        for img in soup.find_all("img"):
            img["loading"] = "lazy"
            img["decoding"] = "async"
            img["class"] = ["report-image"]
            if img.parent.name != "a":
                img.wrap(
                    soup.new_tag(
                        "a",
                        href=img["src"],
                        attrs={
                            "class": "image-link",
                            "aria-label": "打开原图：" + img.get("alt", "实验图片"),
                        },
                    )
                )
        doc["body"] = str(soup)
        if any(
            path == source and note.startswith("已清理")
            for (path, _), note in self.media_notices.items()
        ):
            note_url = self.register_document(
                ROOT / "reports/media-cleanup.md", doc["owner"]
            )
            doc["body"] = (
                f'<aside class="issue-callout"><strong>历史素材已精简</strong><p>本页部分旧录像／帧图已清理，相应链接已标注。数值、日志和失败记录仍保留。见<a href="{note_url}">保留范围与清理记录</a>。</p></aside>'
                + doc["body"]
            )
        doc["toc"] = md.toc
        doc["historical"] = source.name in {
            "P-01-isaaclab.md",
            "P-01-embodichain-dexuni.md",
            "technical-record-v1.md",
        } or source.name in {"asset-preflight.md", "mimic-mr1501-readiness.md"}
        if source.is_relative_to(
            ROOT / "outputs/benchmarks/L-02/reports"
        ) or source.is_relative_to(ROOT / "outputs/benchmarks/L-02/videos"):
            doc["historical"] = True
            doc["body"] = (
                '<aside class="issue-callout"><strong>历史单环境多物体结果</strong><p>该材料不评测 num_envs 多 arena 加载，不用于当前 L-02 的速度或容量结论。</p></aside>'
                + doc["body"]
            )
        self.assets.add(source)
        doc["download"] = self.file_url(source)

    def write_page(self, url: str, template: str, **context: object) -> None:
        path = self.output / unquote(url).lstrip("/")
        if url.endswith("/"):
            path /= "index.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self.templates.get_template(template).render(
                catalog=self.catalog,
                groups=self.groups,
                built_at=self.date,
                counts={
                    key: sum(item["status"] == key for item in self.catalog)
                    for key in STATUS
                },
                **context,
            ),
            encoding="utf-8",
        )

    def build(self) -> dict:
        self.output.mkdir(parents=True, exist_ok=True)
        shutil.copytree(HERE / "static", self.output / "static", dirs_exist_ok=True)
        for item in self.catalog:
            if not item.get("source"):
                continue
            source = ROOT / item["source"]
            self.register_document(source.resolve(), item)
            item["issue"]["url"] = self.local_url(
                item["issue"]["source"], ROOT / "index.md", item
            )
            for resource in item["resources"]:
                resource["url"] = self.local_url(
                    resource["path"], ROOT / "index.md", item
                )
            for video in item["videos"]:
                video["url"] = self.local_url(video["src"], ROOT / "index.md", item)
                video["poster_url"] = self.local_url(
                    video["poster"], ROOT / "index.md", item
                )
            item["thumbnail"] = (
                item["videos"][0]["poster_url"] if item["videos"] else None
            )
        while self.queue:
            self.render_document(self.queue.popleft())

        for path, doc in self.documents.items():
            if path in self.main_sources:
                self.write_page(
                    doc["url"],
                    "experiment.html",
                    item=self.main_sources[path],
                    doc=doc,
                    title=doc["title"],
                )
            else:
                self.write_page(
                    doc["url"],
                    "document.html",
                    item=doc["owner"],
                    doc=doc,
                    title=doc["title"],
                )
        for item in self.catalog:
            if item["status"] == "pending":
                self.write_page(
                    item["url"],
                    "experiment.html",
                    item=item,
                    doc=None,
                    title=f"{item['id']} {item['title']}",
                )
        self.write_page("/", "index.html", title="机器人仿真实验报告", item=None)

        # Only directories explicitly linked by reports are exported. No workspace symlink.
        for directory in list(self.directories):
            for path in directory.rglob("*"):
                resolved = path.resolve()
                if not resolved.is_relative_to(ROOT):
                    raise ValueError(f"Export leaves project: {path}")
                if any(
                    part.startswith(".") for part in path.relative_to(directory).parts
                ):
                    continue
                if path.is_file():
                    self.assets.add(resolved)
                elif path.is_dir():
                    self.directories.add(resolved)
        for directory in sorted(self.directories):
            entries = []
            for path in sorted(
                directory.iterdir(), key=lambda p: (p.is_file(), p.name)
            ):
                if path.resolve() not in self.assets | self.directories:
                    continue
                entries.append(
                    {
                        "name": path.name + ("/" if path.is_dir() else ""),
                        "url": self.file_url(path.resolve())
                        + ("/" if path.is_dir() else ""),
                        "size": (
                            "目录" if path.is_dir() else f"{path.stat().st_size:,} B"
                        ),
                    }
                )
            self.write_page(
                self.file_url(directory) + "/",
                "directory.html",
                title="原始文件目录",
                directory=directory.relative_to(ROOT).as_posix(),
                entries=entries,
                item=None,
            )

        records = []
        for source in sorted(self.assets):
            relative = source.relative_to(ROOT)
            target = self.output / "files" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            records.append(
                {
                    "source": relative.as_posix(),
                    "url": self.file_url(source),
                    "bytes": target.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
        manifest = {
            "built_at": self.date,
            "source_policy": "HTML is a derived reading copy; original reports and raw measurements are unchanged.",
            "experiments": [
                {
                    "id": x["id"],
                    "status": x["status"],
                    "url": x["url"],
                    "source": x.get("source"),
                }
                for x in self.catalog
            ],
            "document_count": len(self.documents),
            "exported_file_count": len(records),
            "exported_bytes": sum(x["bytes"] for x in records),
            "files": records,
            "unpublished_environment_references": self.unpublished,
        }
        (self.output / "publication-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        return {
            key: value
            for key, value in manifest.items()
            if key not in {"files", "experiments", "unpublished_environment_references"}
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/site/public")
    args = parser.parse_args()
    print(json.dumps(SiteBuilder(args.output).build(), ensure_ascii=False, indent=2))
