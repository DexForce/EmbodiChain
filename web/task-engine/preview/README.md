# Offline semantic graph viewer preview

`before.png` uses the previous native SVG viewer; `after.png` uses the unified React viewer. Both display the same frozen E4 graph, recorded report and source-object preview metadata: three stages, fourteen calls and twelve dependencies.

The orange and purple can previews were rendered on the CPU from the original, hash-verified GLBs. They identify source assets; they are not observations after individual calls. These screenshots replay historical evidence and do not qualify a new physical run.

## Standard review SOP

Use this sequence when turning a graph-only benchmark run into a reviewable
offline page. Keep the code snapshot, original inputs, scene declarations and
model configuration frozen for the whole batch.

1. Build the embedded viewer with `npm --prefix web/task-engine run build`.
   This updates the package-local `_graph_react.js` and `_graph_react.css` that
   standalone pages embed; no browser server or network is required.
2. `embodichain gensim task plan` (hybrid route) directly embeds the complete
   review viewer—including `source_image`, `TaskSpec`, role pairing, wrapped
   serpentine layout, and the scientific palette—into
   `PLAN/semantic_task_graph.html`. To re-render a sealed historical plan or
   attach an execution attempt without altering published digests, run
   `embodichain gensim task view --plan PLAN [--run RUN] --output NEW.html`.
   For a batch, use Bash with the campaign directory and a separate output directory:

   ```bash
   campaign=/absolute/path/to/campaign
   destination=/absolute/path/to/previews
   while IFS= read -r -d '' manifest; do
     plan=${manifest%/*}
     case_run=${plan#"$campaign/runs/"}
     html="$destination/${case_run%/plan}/review.html"
     mkdir -p "${html%/*}" || exit 1
     embodichain gensim task view --plan "$plan" --output "$html" || exit 1
   done < <(find "$campaign/runs" -type f -path '*/plan/task_plan_manifest.json' -print0)
   ```

   Keep output outside the sealed plan directories; do not hand-edit their HTML
   or graph JSON.
3. For a benchmark batch, preserve the runner manifest and input hashes, then
   run the fixed RoboDojo/RoboTwin cases three times each. Keep rejected trials
   and never replace them with a later run from another snapshot.
4. Validate every published artifact with `load_task_plan`, graph identity
   checks and source/input hashes. The viewer is evidence for semantic review
   only: `authoring_only` and `execution_ready: false` remain visible.
5. Check the rendered report in Chromium at desktop, short-window and mobile
   widths. Confirm wrapped action cards, milestone/goal links, source image
   preview, role filtering, language/theme controls and keyboard closing still
   work. Record the exact test command and skipped media checks.

This SOP separates presentation quality from semantic acceptance. A polished
page, a `source_reviewed` label or a `covered` graph does not turn an unbound
TaskSpec into a deployment or physical-success certificate.
