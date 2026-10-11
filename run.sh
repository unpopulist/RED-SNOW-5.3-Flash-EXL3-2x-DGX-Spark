#!/bin/bash
# Thin wrapper around the engine fork's deploy scripts (deploy/redsnow in ENGINE_DIR). It adds nothing of its own:
# it points them at this folder's .env and runs them.
#
#   ./run.sh build       # build the image on the head and copy it to the worker (docker/build.sh --to-worker)
#   ./run.sh check       # every launch check, starts nothing (DRY_RUN=1 docker/start-docker.sh)
#   ./run.sh start       # serve in the foreground (run it under tmux); extra arguments go to `tensorfold serve`
#   ./run.sh stop        # stop the pair started by start (docker/stop-docker.sh)
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
export REDSNOW_ENV="${REDSNOW_ENV:-$HERE/.env}"
[[ -f "$REDSNOW_ENV" ]] || { echo "run.sh: no $REDSNOW_ENV (cp .env.example .env and fill it in)" >&2; exit 1; }
if [[ -z "${ENGINE_DIR:-}" ]]; then
  ENGINE_DIR=$(set -a; source "$REDSNOW_ENV"; printf '%s' "${ENGINE_DIR:-}")
fi
[[ -n "$ENGINE_DIR" && "$ENGINE_DIR" != *[\<\>]* ]] || { echo "run.sh: set ENGINE_DIR in $REDSNOW_ENV" >&2; exit 1; }
D="$ENGINE_DIR/deploy/redsnow/docker"
[[ -x "$D/start-docker.sh" ]] || { echo "run.sh: no engine checkout at $ENGINE_DIR (AGENTS.md, step 2)" >&2; exit 1; }

cmd="${1:-}"; shift || true
case "$cmd" in
  build) exec "$D/build.sh" --to-worker "$@" ;;
  check) export DRY_RUN=1; exec "$D/start-docker.sh" "$@" ;;
  start) exec "$D/start-docker.sh" "$@" ;;
  stop)  exec "$D/stop-docker.sh" "$@" ;;
  *) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 2 ;;
esac
