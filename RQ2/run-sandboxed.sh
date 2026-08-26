#!/usr/bin/env bash
set -euo pipefail

BIN="$1"
shift
TIMEOUT="${SANDBOX_TIMEOUT:-2s}"
BACKEND="${SANDBOX_BACKEND:-auto}" # auto|systemd|bwrap

# id="$(uuidgen)"
id=$(date +%s%N | sha256sum | head -c 12)
# Use a dedicated temp dir and clean it even on interruption.
work="$(mktemp -d "/tmp/llm-sbox-$id.XXXXXX")"
mkdir -p "$work/in" "$work/out"
cleanup() { rm -rf "$work"; }
trap cleanup EXIT INT TERM

# Copy binary and all file arguments into the sandbox work dir
cp -a -- "$BIN" "$work/in/prog"
chmod +x "$work/in/prog"

# Copy input files into the sandbox root and rewrite argv to local names.
exec_args=()
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    base="$(basename -- "$arg")"
    cp -a -- "$arg" "$work/$base"
    exec_args+=("$base")
  else
    exec_args+=("$arg")
  fi
done

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
# if [[ "$BACKEND" == "systemd" ]] || { [[ "$BACKEND" == "auto" ]] && can_use_systemd_user; }; then
#   echo "[sandbox] backend=systemd-run" >&2
#   systemd-run --user --unit="llm-sbox-$id" --wait --pipe --quiet \
#     -p RuntimeMaxSec="$TIMEOUT" \
#     -p TimeoutStartSec="$TIMEOUT" \
#     -p TimeoutStopSec="$TIMEOUT" \
#     -p NoNewPrivileges=yes \
#     -p PrivateNetwork=yes \
#     -p PrivateTmp=yes \
#     -p ProtectSystem=strict \
#     -p ProtectHome=yes \
#     -p ProtectKernelTunables=yes \
#     -p ProtectControlGroups=yes \
#     -p RestrictSUIDSGID=yes \
#     -p LockPersonality=yes \
#     -p MemoryDenyWriteExecute=yes \
#     -p RestrictAddressFamilies=AF_UNIX \
#     -p SystemCallFilter='~@mount ~@reboot ~@swap ~@raw-io ~@module ~@debug ~@privileged ~@cpu-emulation' \
#     -p WorkingDirectory="$work" \
#     -p ReadWritePaths="$work" \
#     /bin/bash -lc './in/prog "$@" >out/stdout 2>out/stderr; rc=$?; echo $rc >out/exitcode; exit $rc' -- "${exec_args[@]}"
#   status=$?
if [[ "$BACKEND" == "bwrap" ]] || { [[ "$BACKEND" == "auto" ]] && command -v bwrap >/dev/null 2>&1; }; then
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
else
  echo "No supported sandbox backend found (systemd-run or bwrap)." >&2
  status=127
fi

set -e

# Preserve the program exit code if captured.
if [[ -f "$work/out/exitcode" ]]; then
  status="$(cat "$work/out/exitcode" 2>/dev/null || echo "$status")"
fi

# Copy last arg back if it looks like an output filename produced in sandbox.
if [[ ${#exec_args[@]} -gt 0 ]]; then
  last_arg="${exec_args[${#exec_args[@]}-1]}"
  if [[ "$last_arg" != -* ]] && [[ "$last_arg" != */* ]] && [[ -f "$work/$last_arg" ]]; then
    cp -a -- "$work/$last_arg" "./$last_arg"
  fi
fi

echo "--- stdout ---"
cat "$work/out/stdout" 2>/dev/null || true

# If it failed, show us why
if [ $status -ne 0 ]; then
  echo "--- sandbox stderr ---" >&2
  cat "$work/out/stderr" >&2
fi

exit "$status"
