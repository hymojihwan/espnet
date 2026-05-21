#!/usr/bin/env bash
set -e
set -u
set -o pipefail

script_dir="$(cd "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${script_dir}"

ENH_CONFIG=conf/tuning/train_enh_convtasnet_small.yaml ./run.sh "$@"
