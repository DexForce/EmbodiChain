#!/usr/bin/env python3
"""
Post-build script for sphinx-multiversion to create a root index with version redirect.

This script:
1. Finds the latest version (latest tag or main branch)
2. Creates a root index.html that redirects to the latest version
3. Optionally adds a version selector
"""

import os
import shutil
import re
from pathlib import Path


def get_versions(html_dir: Path) -> list[tuple[str, str]]:
    """Get list of versions from the built HTML directory."""
    versions = []
    if not html_dir.exists():
        return versions

    for item in html_dir.iterdir():
        if item.is_dir() and item.name != "_static" and item.name != "_sources":
            # Check if it looks like a version (tag or branch)
            if re.match(r"^v\d+\.\d+\.\d+$", item.name) or item.name in ("main", "dev"):
                versions.append((item.name, str(item)))

    # Sort versions - latest tag first, then branches
    def version_sort(v):
        name = v[0]
        if name.startswith("v"):
            # Parse semantic version
            parts = name[1:].split(".")
            return (0, [-int(p) for p in parts])  # Higher versions first
        else:
            # Branches second, main first
            return (1, [0 if name == "main" else 1])

    versions.sort(key=version_sort)
    return versions


def create_redirect_index(html_dir: Path, versions: list[tuple[str, str]]) -> None:
    """Create a root index.html that redirects to the latest version."""
    if not versions:
        print("No versions found, skipping redirect creation")
        return

    latest_version = versions[0][0]
    # Root index is in build/html/, versions are in build/html/docs/
    redirect_url = f"docs/{latest_version}/index.html"

    # Read the existing index from latest version as template
    latest_index = html_dir / latest_version / "index.html"
    if latest_index.exists():
        with open(latest_index, "r") as f:
            content = f.read()
    else:
        # Fallback: create simple redirect
        content = None

    # Root index goes in build/html/, not build/html/docs/
    root_index = html_dir.parent / "index.html"

    # Create a simple HTML that redirects with a version selector
    redirect_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EmbodiChain Documentation</title>
    <meta http-equiv="refresh" content="0;url={redirect_url}">
    <link rel="stylesheet" href="_static/css/theme.css">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background: #f5f5f5;
        }}
        .container {{
            text-align: center;
            padding: 2rem;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; margin-bottom: 1rem; }}
        .version-info {{ color: #666; margin-bottom: 1.5rem; }}
        .btn {{
            display: inline-block;
            padding: 0.75rem 1.5rem;
            background: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            margin: 0.25rem;
        }}
        .btn:hover {{ background: #0056b3; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>EmbodiChain Documentation</h1>
        <p class="version-info">Redirecting to version: <strong>{latest_version}</strong></p>
        <div>
            <a href="{redirect_url}" class="btn">Go to Latest ({latest_version})</a>
        </div>
        <p style="margin-top: 2rem; color: #888; font-size: 0.9rem;">
            Other versions:
"""

    # Add links to other versions
    for version_name, _ in versions[1:6]:  # Show up to 5 other versions
        version_url = f"docs/{version_name}/index.html"
        redirect_html += f'\n            <a href="{version_url}" style="color: #007bff;">{version_name}</a>'

    redirect_html += (
        """
        </p>
        <p style="margin-top: 1rem; color: #888; font-size: 0.8rem;">
            If you are not redirected, <a href="#" onclick="window.location.href=document.querySelector('.btn').href">click here</a>.
        </p>
    </div>
    <script>
        // Auto-redirect after a short delay
        setTimeout(function() {
            window.location.href = "docs/"""
        + latest_version
        + """/index.html";
        }, 1500);
    </script>
</body>
</html>
"""
    )

    with open(root_index, "w") as f:
        f.write(redirect_html)

    print(f"Created redirect index.html -> {redirect_url}")


def copy_latest_to_root(html_dir: Path, versions: list[tuple[str, str]]) -> None:
    """Copy latest version content to root for direct access."""
    if not versions:
        return

    latest_version = versions[0][0]
    latest_dir = html_dir / latest_version
    root_dir = html_dir.parent

    if not latest_dir.exists():
        print(f"Latest version directory not found: {latest_dir}")
        return

    # Copy latest version files to root for direct access
    # This allows accessing docs without version prefix
    for item in latest_dir.iterdir():
        if item.name == "index.html":
            continue  # Skip, we use our redirect
        dest = root_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    print(f"Copied latest version ({latest_version}) files to root")


def main():
    # The smv_outputdir_format = "docs/{ref.name}" puts versions in build/html/docs/
    html_dir = Path(__file__).parent.parent / "build" / "html" / "docs"

    print(f"Processing versions in: {html_dir}")

    # Get all versions
    versions = get_versions(html_dir)
    print(f"Found versions: {[v[0] for v in versions]}")

    if versions:
        create_redirect_index(html_dir, versions)
        copy_latest_to_root(html_dir, versions)
    else:
        print("No versioned docs found. Run 'make multiview' first.")


if __name__ == "__main__":
    main()
