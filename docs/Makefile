# Makefile for Sphinx documentation with multiversion support
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
VERSIONDIR    = build/html

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@rm -rf "$(BUILDDIR)"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for multiversion builds
.PHINXFLAGS  ?=

.PHONY: multiview clean-multiview

# Build multiversion docs (incremental - only rebuilds changed versions)
multiview:
	sphinx-multiversion "$(SOURCEDIR)" "$(VERSIONDIR)" $(SPHINXOPTS) $(O)
	@python scripts/post_build_multiversion.py

# Force full rebuild of all versions (use when cache is corrupted)
multiview-full:
	@rm -rf "$(BUILDDIR)"
	sphinx-multiversion "$(SOURCEDIR)" "$(VERSIONDIR)" $(SPHINXOPTS) $(O)
	@python scripts/post_build_multiversion.py

# Build only specific tags/branches
multiview-tag:
	@rm -rf "$(BUILDDIR)"
	sphinx-multiversion "$(SOURCEDIR)" "$(VERSIONDIR)" \
		-D smv_tag_whitelist="^v\d+\.\d+\.\d+$" \
		-D smv_branch_whitelist="^(main|dev)$$" \
		$(SPHINXOPTS) $(O)

# Clean build artifacts
clean-multiview:
	@rm -rf "$(BUILDDIR)"
