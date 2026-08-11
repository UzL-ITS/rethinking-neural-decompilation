#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CALLER_CWD="$(pwd)"

BIN="$1"
shift

TIMEOUT="${SANDBOX_TIMEOUT:-2s}"
BACKEND="${SANDBOX_BACKEND:-systemd}"   # systemd|bwrap|auto|none

id="$(date +%s%N | sha256sum | head -c 12)"
work="$(mktemp -d "/tmp/llm-sbox-$id.XXXXXX")"
mkdir -p "$work/in" "$work/out"

cleanup() { rm -rf "$work"; }
trap cleanup EXIT INT TERM

if [[ "$BIN" = /* ]]; then
  BIN_PATH="$BIN"
else
  BIN_PATH="$CALLER_CWD/$BIN"
fi

cp -a -- "$BIN_PATH" "$work/in/prog"
chmod +x "$work/in/prog"

# Copy file arguments into the sandbox root and rewrite argv to local names.
exec_args=()
last_output_host_path=""
last_output_sandbox_name=""
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    base="$(basename -- "$arg")"
    if [[ "$arg" = /* ]]; then
      ARG_PATH="$arg"
    else
      ARG_PATH="$CALLER_CWD/$arg"
    fi
    cp -a -- "$ARG_PATH" "$work/$base"
    exec_args+=("$base")
  else
    exec_args+=("$arg")
  fi
done

# For ExeBench we commonly call:
#   run-sandboxed.sh ./test input_tmp.json output_tmp.json
# The last non-binary argument is typically the output JSON file. If it does
# not exist yet, create an empty placeholder in the sandbox and copy it back.
if [[ $# -ge 2 ]]; then
  candidate_output="${@: -1}"
  if [[ "$candidate_output" != -* ]]; then
    last_output_host_path="$candidate_output"
    last_output_sandbox_name="$(basename -- "$candidate_output")"
    : > "$work/$last_output_sandbox_name"
  fi
fi

run_with_timeout() {
  if command -v timeout >/dev/null 2>&1; then
    timeout "$TIMEOUT" "$@"
  else
    "$@"
  fi
}

can_use_systemd_user() {
  command -v systemd-run >/dev/null 2>&1 || return 1
  systemd-run --user --wait --pipe --quiet /bin/true >/dev/null 2>&1
}

set +e
if [[ "$BACKEND" == "systemd" ]] || { [[ "$BACKEND" == "auto" ]] && can_use_systemd_user; }; then
  echo "[sandbox] backend=systemd-run" >&2
  systemd-run --user --unit="llm-sbox-$id" --wait --pipe --quiet \
    -p RuntimeMaxSec="$TIMEOUT" \
    -p TimeoutStartSec="$TIMEOUT" \
    -p TimeoutStopSec="$TIMEOUT" \
    -p NoNewPrivileges=yes \
    -p PrivateNetwork=yes \
    -p PrivateTmp=yes \
    -p ProtectSystem=strict \
    -p ProtectHome=yes \
    -p ProtectKernelTunables=yes \
    -p ProtectControlGroups=yes \
    -p RestrictSUIDSGID=yes \
    -p LockPersonality=yes \
    -p MemoryDenyWriteExecute=yes \
    -p RestrictAddressFamilies=AF_UNIX \
    -p SystemCallFilter='~@mount ~@reboot ~@swap ~@raw-io ~@module ~@debug ~@privileged ~@cpu-emulation' \
    -p WorkingDirectory="$work" \
    -p ReadWritePaths="$work" \
    /bin/bash -lc './in/prog "$@" >out/stdout 2>out/stderr; rc=$?; echo $rc >out/exitcode; exit $rc' -- "${exec_args[@]}"
  status=$?
elif [[ "$BACKEND" == "bwrap" ]] || { [[ "$BACKEND" == "auto" ]] && command -v bwrap >/dev/null 2>&1; }; then
  echo "[sandbox] backend=bwrap" >&2
  run_with_timeout bwrap \
    --die-with-parent \
    --new-session \
    --unshare-ipc \
    --unshare-net \
    --unshare-pid \
    --unshare-uts \
    --unshare-cgroup-try \
    --cap-drop ALL \
    --clearenv \
    --setenv PATH /usr/sbin:/usr/bin:/sbin:/bin \
    --setenv HOME /tmp \
    --setenv LANG C.UTF-8 \
    --ro-bind / / \
    --proc /proc \
    --dev /dev \
    --tmpfs /tmp \
    --tmpfs /run \
    --tmpfs /var/tmp \
    --tmpfs /home \
    --dir /tmp/work \
    --bind "$work" /tmp/work \
    --chdir /tmp/work \
    /bin/bash -lc './in/prog "$@" >out/stdout 2>out/stderr; rc=$?; echo $rc >out/exitcode; exit $rc' -- "${exec_args[@]}"
  status=$?
elif [[ "$BACKEND" == "none" ]]; then
  echo "[sandbox] backend=none (direct execution fallback)" >&2
  (
    cd "$work"
    /bin/bash -lc './in/prog "$@" >out/stdout 2>out/stderr; rc=$?; echo $rc >out/exitcode; exit $rc' -- "${exec_args[@]}"
  )
  status=$?
else
  echo "No supported sandbox backend found (systemd-run or bwrap), or requested backend unavailable." >&2
  status=127
fi
set -e

if [[ -f "$work/out/exitcode" ]]; then
  status="$(cat "$work/out/exitcode" 2>/dev/null || echo "$status")"
fi

if [[ -n "$last_output_host_path" ]] && [[ -f "$work/$last_output_sandbox_name" ]]; then
  if [[ "$last_output_host_path" = /* ]]; then
    OUTPUT_PATH="$last_output_host_path"
  else
    OUTPUT_PATH="$CALLER_CWD/$last_output_host_path"
  fi
  cp -a -- "$work/$last_output_sandbox_name" "$OUTPUT_PATH"
fi

echo "--- stdout ---"
cat "$work/out/stdout" 2>/dev/null || true

if [[ "$status" -ne 0 ]]; then
  echo "--- sandbox stderr ---" >&2
  cat "$work/out/stderr" 2>/dev/null >&2 || true
fi

exit "$status"
