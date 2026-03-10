#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RAW_DIR="$ROOT_DIR/bin/raw"

usage() {
  cat <<'EOF'
Usage: scripts/assemble-riscv.sh [input.s] [output.bin]

Without arguments, assembles every .s/.S file in bin/raw into a matching .bin.
With one argument, assembles that source file into a sibling .bin.
With two arguments, assembles the input source file into the given output path.

Environment:
  RISCV_PREFIX   Toolchain prefix to use. Examples:
                 riscv64-unknown-elf
                 riscv64-linux-gnu
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 2 ]]; then
  usage >&2
  exit 1
fi

toolchain_prefixes=()
if [[ -n "${RISCV_PREFIX:-}" ]]; then
  toolchain_prefixes+=("$RISCV_PREFIX")
fi
toolchain_prefixes+=("riscv64-unknown-elf" "riscv64-linux-gnu")

gcc_bin=""
objcopy_bin=""

for prefix in "${toolchain_prefixes[@]}"; do
  if command -v "${prefix}-gcc" >/dev/null 2>&1 && command -v "${prefix}-objcopy" >/dev/null 2>&1; then
    gcc_bin="${prefix}-gcc"
    objcopy_bin="${prefix}-objcopy"
    break
  fi
done

if [[ -z "$gcc_bin" || -z "$objcopy_bin" ]]; then
  cat >&2 <<'EOF'
error: no supported RISC-V GNU toolchain found.

Install one of:
  - riscv64-unknown-elf-gcc + riscv64-unknown-elf-objcopy
  - riscv64-linux-gnu-gcc + riscv64-linux-gnu-objcopy

Or set RISCV_PREFIX to a matching toolchain prefix.
EOF
  exit 1
fi

assemble_one() {
  local input=$1
  local output=$2
  local tmp_elf

  if [[ ! -f "$input" ]]; then
    echo "error: input file not found: $input" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$output")"

  tmp_elf=$(mktemp /tmp/rvemu-asm.XXXXXX.elf)
  "$gcc_bin" \
    -nostdlib \
    -nostartfiles \
    -static \
    -Wl,-Ttext=0x80000000 \
    -Wl,--build-id=none \
    -o "$tmp_elf" \
    "$input"

  "$objcopy_bin" -O binary "$tmp_elf" "$output"
  rm -f "$tmp_elf"

  echo "assembled $input -> $output"
}

if [[ $# -eq 0 ]]; then
  shopt -s nullglob
  inputs=("$RAW_DIR"/*.s "$RAW_DIR"/*.S)
  shopt -u nullglob

  if [[ ${#inputs[@]} -eq 0 ]]; then
    echo "error: no assembly files found in $RAW_DIR" >&2
    exit 1
  fi

  for input in "${inputs[@]}"; do
    assemble_one "$input" "${input%.*}.bin"
  done
elif [[ $# -eq 1 ]]; then
  assemble_one "$1" "${1%.*}.bin"
else
  assemble_one "$1" "$2"
fi
