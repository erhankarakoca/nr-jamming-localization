#!/usr/bin/env bash
# Build both drafts from draft.tex: draft-A.pdf (receiver story) and draft-C.pdf (both stories).
# The figures come from the repo's own figure code:
#   JAMLOC_RESULTS_DIR=$PWD/build python -m jamloc.experiments.figures \
#       --only attack_patterns residual_explained
# and are copied to figures/ (build/ is gitignored).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build/tex

for story in A C; do
  job="draft-${story}"
  if [ "$story" = A ]; then preamble='\def\storyA{}'; else preamble=''; fi
  for pass in 1 2 3; do
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build/tex \
      -jobname="$job" "${preamble}\input{draft.tex}" > "build/tex/${job}.pass${pass}.log"
    if [ "$pass" = 1 ]; then
      # references.bib lives in the repo's docs/ folder, five levels up from build/tex
      (cd build/tex && BIBINPUTS=../../../../../docs: bibtex "$job" > "${job}.bibtex.log") || true
    fi
  done
  cp "build/tex/${job}.pdf" "${job}.pdf"
  echo "Wrote ${job}.pdf"
done
