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

"""Check exported links, source integrity, HTTP serving and video byte ranges."""

from __future__ import annotations

import os

import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("SIM_REPORT_ROOT", HERE.parents[4])).resolve()


def check(public: Path, base_url: str) -> dict:
    """Validate actual exported artifacts; no simulator is started."""
    manifest = json.loads((public / "publication-manifest.json").read_text())
    generated = [
        public / "index.html",
        *sorted((public / "experiments").glob("*.html")),
        *sorted((public / "notes").glob("*.html")),
    ]
    generated.extend(
        p
        for p in (public / "files").rglob("index.html")
        if p.relative_to(public).as_posix()
        not in {"files/" + r["source"] for r in manifest["files"]}
    )
    pages = {path: BeautifulSoup(path.read_text(), "html.parser") for path in generated}
    broken = []
    fragments = []
    link_count = 0
    for path, soup in pages.items():
        for element in soup.select(
            "a[href], img[src], source[src], video[poster], script[src], link[href]"
        ):
            reference = (
                element.get("href") or element.get("src") or element.get("poster")
            )
            parsed = urlsplit(reference)
            if parsed.scheme or parsed.netloc:
                continue
            target = (
                public / unquote(parsed.path).lstrip("/")
                if parsed.path.startswith("/")
                else path.parent / unquote(parsed.path)
            )
            target = target if parsed.path else path
            if target.is_dir():
                target /= "index.html"
            target = target.resolve()
            link_count += 1
            if not target.is_relative_to(public) or not target.is_file():
                broken.append([str(path.relative_to(public)), reference])
            if (
                parsed.fragment
                and target in pages
                and not pages[target].find(id=unquote(parsed.fragment))
            ):
                fragments.append([str(path.relative_to(public)), reference])
    assert not broken, json.dumps({"broken_links": broken}, ensure_ascii=False)
    assert not fragments, json.dumps(
        {"broken_fragments": fragments}, ensure_ascii=False
    )
    assert len(list((public / "experiments").glob("*.html"))) == 13
    catalog = json.loads((HERE / "catalog.json").read_text())
    assert {x["id"]: x["status"] for x in manifest["experiments"]} == {
        x["id"]: x["status"] for x in catalog
    }
    for item in manifest["experiments"]:
        soup = pages[public / item["url"].lstrip("/")]
        assert (
            soup.select_one(".page-header .badge").get_text()
            == {"complete": "本轮完成", "partial": "部分完成", "pending": "待开展"}[
                item["status"]
            ]
        )
        if item["status"] == "pending":
            assert "内容待补充" in soup.get_text() and not soup.select(
                "video, .metrics"
            )

    for record in manifest["files"]:
        for path in [ROOT / record["source"], public / "files" / record["source"]]:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            assert digest.hexdigest() == record["sha256"], str(path)

    checks = []
    for path in [
        "/",
        *[x["url"] for x in manifest["experiments"]],
        "/static/site.css",
        "/static/site.js",
        "/publication-manifest.json",
    ]:
        with urlopen(base_url + path, timeout=10) as response:
            assert response.status == 200
            checks.append({"url": path, "status": response.status})
    for path in [
        "/AGENTS.md",
        "/.git/config",
        "/../README.md",
        "/%2e%2e/README.md",
        "/files/../../../../etc/passwd",
        "/experiments/nonexistent.html",
    ]:
        try:
            urlopen(base_url + path, timeout=10)
            raise AssertionError(f"Unexpected accessible path: {path}")
        except HTTPError as error:
            assert error.code == 404, (path, error.code)
    try:
        urlopen(Request(base_url + "/", method="POST", data=b"test"), timeout=10)
        raise AssertionError("Server accepted a write method")
    except HTTPError as error:
        assert error.code == 405
    media = []
    for path, soup in pages.items():
        for source in soup.select("video source[src]"):
            url = source["src"]
            with urlopen(
                Request(base_url + url, headers={"Range": "bytes=0-1023"}), timeout=10
            ) as response:
                assert response.status == 206
                assert response.headers["Content-Type"].startswith("video/mp4")
                assert response.headers["Content-Range"].startswith("bytes 0-1023/")
                assert len(response.read()) == 1024
            with urlopen(
                Request(base_url + url, method="HEAD"), timeout=10
            ) as response:
                assert (
                    response.status == 200
                    and int(response.headers["Content-Length"]) > 1024
                )
            media.append(url)
    return {
        "pages_checked": len(pages),
        "links_checked": link_count,
        "source_files_verified": len(manifest["files"]),
        "video_range_checks": len(media),
        "http_checks": checks,
        "broken_links": broken,
        "broken_fragments": fragments,
        "read_only_routes": True,
        "source_hashes_match": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "outputs/site/public")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    result = check(args.directory.resolve(), args.url.rstrip("/"))
    (args.directory.parent / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
