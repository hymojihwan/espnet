#!/usr/bin/env python3
"""
Create CHiME4 tr05_org WAV files from WSJ0 SPHERE (.wv1) and tr05_org.json.

CHiME4 annotations provide tr05_org.json but the actual audio is usually from
the LDC CHiME-3 package. If you only have CHiME4 annotations + WSJ0 (e.g.
CHiME4/data/WSJ0 with si_tr_s in SPHERE format), this script creates
tr05_org/*.wav so that the MATLAB simulation can run.

Usage:
  python3 local/create_tr05_org_from_wsj0.py <CHIME4_root> [<WSJ0_root>]

  CHIME4_root: e.g. /DB/CHiME4 (must contain data/annotations/tr05_org.json
               and data/audio/16kHz/isolated/tr05_org will be created here)
  WSJ0_root:   e.g. /DB/CHiME4/data/WSJ0 or /DB/WSJ0 (default: CHIME4_root/data/WSJ0)
               Must contain wsj0/si_tr_s/<speaker>/<speaker><wsjname>.wv1
"""

import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Create tr05_org WAVs from WSJ0 + tr05_org.json")
    parser.add_argument("chime4_root", help="CHiME4 root (e.g. /DB/CHiME4)")
    parser.add_argument("wsj0_root", nargs="?", default=None,
                        help="WSJ0 root (default: CHIME4_root/data/WSJ0)")
    args = parser.parse_args()

    chime4 = os.path.abspath(args.chime4_root)
    wsj0 = os.path.abspath(args.wsj0_root) if args.wsj0_root else os.path.join(chime4, "data", "WSJ0")
    ann_file = os.path.join(chime4, "data", "annotations", "tr05_org.json")
    out_dir = os.path.join(chime4, "data", "audio", "16kHz", "isolated", "tr05_org")

    if not os.path.isfile(ann_file):
        print(f"Error: annotations not found: {ann_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(wsj0):
        print(f"Error: WSJ0 directory not found: {wsj0}", file=sys.stderr)
        sys.exit(1)

    # Find sph2pipe (same as in Kaldi/ESPnet)
    sph2pipe = "sph2pipe"
    if subprocess.run(["which", sph2pipe], capture_output=True).returncode != 0:
        print("Error: sph2pipe not found in PATH. Install Kaldi or add sph2pipe to PATH.", file=sys.stderr)
        sys.exit(1)

    with open(ann_file, "r", encoding="utf-8") as f:
        utts = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    missing = []
    for u in utts:
        speaker = u["speaker"]
        wsj_name = u["wsj_name"]
        # WSJ0 path: wsj0/si_tr_s/<speaker>/<speaker><wsjname_lower>.wv1
        wsj_lower = wsj_name.lower()
        rel_path = os.path.join("wsj0", "si_tr_s", speaker, f"{speaker}{wsj_lower}.wv1")
        sph_path = os.path.join(wsj0, rel_path)
        out_name = f"{speaker}_{wsj_name}_ORG.wav"
        out_path = os.path.join(out_dir, out_name)

        if os.path.exists(out_path):
            continue
        if not os.path.isfile(sph_path):
            missing.append(sph_path)
            continue
        # Convert to 16 kHz WAV
        subprocess.run(
            [sph2pipe, "-f", "wav", "-p", "-c", "1", sph_path, out_path],
            check=True,
            capture_output=True,
        )

    if missing:
        print(f"Warning: {len(missing)} WSJ0 files not found (first few):", file=sys.stderr)
        for p in missing[:5]:
            print(f"  {p}", file=sys.stderr)
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more", file=sys.stderr)
    print(f"Created tr05_org WAVs in {out_dir}")


if __name__ == "__main__":
    main()
