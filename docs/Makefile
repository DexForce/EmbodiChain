# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@rm -rf "$(BUILDDIR)"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Multi-version build target for local testing
.PHONY: multiversion
multiversion:
	@rm -rf "$(BUILDDIR)/html"
	@echo "Building multi-version documentation..."
	@sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "Generating versions.json..."
	@python3 scripts/generate_versions_json.py --build-dir "$(BUILDDIR)/html" --output "$(BUILDDIR)/html/versions.json"
	@echo "Creating root index.html with inline redirect..."
	@cp "$(SOURCEDIR)/_templates/index.html" "$(BUILDDIR)/html/index.html" 2>/dev/null || echo '<meta http-equiv="refresh" content="0; url=main/">' > "$(BUILDDIR)/html/index.html"
	@echo "Multi-version docs built successfully."

# Clean multi-version build artifacts
.PHONY: clean-multiversion
clean-multiversion:
	@rm -rf "$(BUILDDIR)/html"
	@echo "Multi-version build artifacts cleaned."

# Preview documentation with local HTTP server
.PHONY: preview
preview:
	@echo "Starting documentation preview server at http://localhost:8000"
	@python3 preview.py --directory "$(BUILDDIR)/html" --port 8000 --open
