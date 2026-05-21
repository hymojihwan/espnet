#!/usr/bin/env bash
set -e
set -u
set -o pipefail

script_dir="$(cd "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
exec "${script_dir}/run_enh_s2t_frozen_enh_ctc.sh" "$@"
