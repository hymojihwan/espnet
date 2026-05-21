#!/usr/bin/env bash

# Copyright 2014  Vassil Panayotov
#           2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <src-dir> <dst-dir>"
  echo "e.g.: $0 /export/a15/vpanayotov/data/LibriSpeech/dev-clean data/dev-clean"
  exit 1
fi

src=$1
dst=$2

# all utterances are FLAC compressed
if ! which flac >&/dev/null; then
   echo "Please install 'flac' on ALL worker nodes!"
   exit 1
fi

spk_file=$src/../SPEAKERS.TXT

mkdir -p $dst || exit 1

[ ! -d $src ] && echo "$0: no such directory $src" && exit 1

# Handle missing SPEAKERS.TXT file
if [ ! -f $spk_file ]; then
    echo "$0: SPEAKERS.TXT not found at $spk_file, creating default one"
    # Create a default SPEAKERS.TXT with all speakers as male (m)
    # This is a fallback for datasets that don't have speaker gender information
    echo "# Default SPEAKERS.TXT created for dataset without speaker gender information" > $spk_file
    echo "# All speakers are set to male (m) as default" >> $spk_file
    echo "# Format: ID | Gender | Name | Set | Description" >> $spk_file
fi

wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
trans=$dst/text; [[ -f "$trans" ]] && rm $trans
utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk
spk2gender=$dst/spk2gender; [[ -f $spk2gender ]] && rm $spk2gender

# Also try to load database paths to create clean references
if [ -f ./db.sh ]; then
  . ./db.sh
fi

for reader_dir in $(find -L $src -mindepth 1 -maxdepth 1 -type d | sort); do
  reader=$(basename $reader_dir)
  if ! [ $reader -eq $reader ]; then  # not integer.
    echo "$0: unexpected subdirectory name $reader"
    exit 1
  fi

  # Try to get gender from SPEAKERS.TXT, default to 'm' if not found
  reader_gender=$(egrep "^$reader[ ]+\|" $spk_file 2>/dev/null | awk -F'|' '{gsub(/[ ]+/, ""); print tolower($2)}' || echo "m")
  if [ "$reader_gender" != 'm' ] && [ "$reader_gender" != 'f' ]; then
    echo "Unexpected gender: '$reader_gender', defaulting to 'm'"
    reader_gender="m"
  fi

  for chapter_dir in $(find -L $reader_dir/ -mindepth 1 -maxdepth 1 -type d | sort); do
    chapter=$(basename $chapter_dir)
    if ! [ "$chapter" -eq "$chapter" ]; then
      echo "$0: unexpected chapter-subdirectory name $chapter"
      exit 1
    fi

    find -L $chapter_dir/ -iname "*.flac" | sort | xargs -I% basename % .flac | \
      awk -v "dir=$chapter_dir" '{printf "%s %s/%s.flac\n", $0, dir, $0}' >>$wav_scp|| exit 1

    chapter_trans=$chapter_dir/${reader}-${chapter}.trans.txt
    [ ! -f  $chapter_trans ] && echo "$0: expected file $chapter_trans to exist" && exit 1
    cat $chapter_trans >>$trans

    # NOTE: For now we are using per-chapter utt2spk. That is each chapter is considered
    #       to be a different speaker. This is done for simplicity and because we want
    #       e.g. the CMVN to be calculated per-chapter
    awk -v "reader=$reader" -v "chapter=$chapter" '{printf "%s %s-%s\n", $1, reader, chapter}' \
      <$chapter_trans >>$utt2spk || exit 1

    # reader -> gender map (again using per-chapter granularity)
    echo "${reader}-${chapter} $reader_gender" >>$spk2gender
  done
done

spk2utt=$dst/spk2utt
utils/utt2spk_to_spk2utt.pl <$utt2spk >$spk2utt || exit 1

# If this dataset comes from NOISY_LIBRISPEECH, generate spk1.scp pointing to clean LibriSpeech
if [ -n "${NOISY_LIBRISPEECH:-}" ] && [ -n "${LIBRISPEECH:-}" ]; then
  if [[ "${src}" == ${NOISY_LIBRISPEECH}/* ]]; then
    part=$(basename "${src}")
    clean_prefix="${LIBRISPEECH}/LibriSpeech/${part}"
    spk1_scp=$dst/spk1.scp; [[ -f "$spk1_scp" ]] && rm "$spk1_scp"
    # Map each noisy wav path back to the clean LibriSpeech path by replacing the src prefix
    while read -r utt path; do
      rel_path="${path#${src}/}"
      echo "${utt} ${clean_prefix}/${rel_path}"
    done < "$wav_scp" | sort > "$spk1_scp"
  fi
fi

ntrans=$(wc -l <$trans)
nutt2spk=$(wc -l <$utt2spk)
! [ "$ntrans" -eq "$nutt2spk" ] && \
  echo "Inconsistent #transcripts($ntrans) and #utt2spk($nutt2spk)" && exit 1

utils/validate_data_dir.sh --no-feats $dst || exit 1

echo "$0: successfully prepared data in $dst"

exit 0 