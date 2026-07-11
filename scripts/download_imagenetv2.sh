#!/usr/bin/env bash
set -euo pipefail

# Download ImageNet-V2 MatchedFrequency for this repo.
# This project expects the extracted folder to be:
#   ${DATA_ROOT}/imagenetv2-matched-frequency-format-val
#
# Usage:
#   bash scripts/download_imagenetv2.sh /path/to/data/root
#   bash scripts/download_imagenetv2.sh /path/to/data/root --url https://your-mirror/imagenetv2-matched-frequency.tar.gz
#
# Environment variables:
#   IMAGENETV2_URL       Override download URL.
#   IMAGENETV2_KEEP_TAR  Set to 1 to keep the .tar.gz after extraction.

DATA_ROOT="${1:-}"
if [[ -z "${DATA_ROOT}" || "${DATA_ROOT}" == "-h" || "${DATA_ROOT}" == "--help" ]]; then
  sed -n '1,18p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi
shift || true

CUSTOM_URL="${IMAGENETV2_URL:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)
      CUSTOM_URL="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

TARGET_NAME="imagenetv2-matched-frequency-format-val"
TARGET_DIR="${DATA_ROOT}/${TARGET_NAME}"
ARCHIVE="${DATA_ROOT}/imagenetv2-matched-frequency.tar.gz"
TMP_EXTRACT="${DATA_ROOT}/.imagenetv2_extract_tmp"

# Put domestic-friendly mirrors first. hf-mirror.com is commonly reachable from mainland China.
URLS=()
if [[ -n "${CUSTOM_URL}" ]]; then
  URLS+=("${CUSTOM_URL}")
fi
URLS+=(
  "https://hf-mirror.com/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz"
  "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz"
  "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz"
)

mkdir -p "${DATA_ROOT}"

if [[ -d "${TARGET_DIR}" ]]; then
  echo "[OK] Target already exists: ${TARGET_DIR}"
  echo "You can run: python tpt_classification.py ${DATA_ROOT} --test_sets V -a RN50 -b 8 --gpu 0 --tpt --ctx_init a_photo_of_a"
  exit 0
fi

if [[ ! -s "${ARCHIVE}" ]]; then
  echo "Downloading ImageNet-V2 MatchedFrequency to: ${ARCHIVE}"
  success=0
  for url in "${URLS[@]}"; do
    echo "Trying: ${url}"
    if command -v aria2c >/dev/null 2>&1; then
      if aria2c -x 8 -s 8 -k 1M --continue=true --auto-file-renaming=false \
          --dir "${DATA_ROOT}" --out "$(basename "${ARCHIVE}")" "${url}"; then
        success=1
        break
      fi
    elif command -v wget >/dev/null 2>&1; then
      if wget -c -O "${ARCHIVE}" "${url}"; then
        success=1
        break
      fi
    elif command -v curl >/dev/null 2>&1; then
      if curl -L --fail --retry 3 -C - -o "${ARCHIVE}" "${url}"; then
        success=1
        break
      fi
    else
      echo "Need one of: aria2c, wget, curl" >&2
      exit 1
    fi
    echo "Failed: ${url}"
  done
  if [[ "${success}" != "1" ]]; then
    echo "All download URLs failed. Try a custom mirror:" >&2
    echo "  bash scripts/download_imagenetv2.sh ${DATA_ROOT} --url https://.../imagenetv2-matched-frequency.tar.gz" >&2
    exit 1
  fi
else
  echo "[OK] Archive already exists: ${ARCHIVE}"
fi

rm -rf "${TMP_EXTRACT}"
mkdir -p "${TMP_EXTRACT}"
echo "Extracting..."
# The official file is named .tar.gz, but some mirrors may serve it as a
# plain tar archive. Try gzip first, then fall back to regular tar.
if gzip -t "${ARCHIVE}" >/dev/null 2>&1; then
  tar -xzf "${ARCHIVE}" -C "${TMP_EXTRACT}"
else
  echo "Archive is not gzip-compressed; trying plain tar extraction..."
  if ! tar -xf "${ARCHIVE}" -C "${TMP_EXTRACT}"; then
    echo "Extraction failed. File type:" >&2
    file "${ARCHIVE}" >&2 || true
    echo "If this says HTML/text, delete the archive and retry with another URL:" >&2
    echo "  rm -f ${ARCHIVE}" >&2
    echo "  bash scripts/download_imagenetv2.sh ${DATA_ROOT} --url https://.../imagenetv2-matched-frequency.tar.gz" >&2
    exit 1
  fi
fi

# Normalize extracted folder name to what data/datautils.py expects.
FOUND=""
for candidate in \
  "${TMP_EXTRACT}/imagenetv2-matched-frequency-format-val" \
  "${TMP_EXTRACT}/imagenetv2-matched-frequency" \
  "${TMP_EXTRACT}/matched-frequency"; do
  if [[ -d "${candidate}" ]]; then
    FOUND="${candidate}"
    break
  fi
done

# Some archives may contain class folders directly at archive root.
if [[ -z "${FOUND}" && -d "${TMP_EXTRACT}/0" && -d "${TMP_EXTRACT}/999" ]]; then
  FOUND="${TMP_EXTRACT}"
fi

if [[ -z "${FOUND}" ]]; then
  echo "Could not find extracted ImageNet-V2 folder under ${TMP_EXTRACT}" >&2
  echo "Top-level extracted entries:" >&2
  find "${TMP_EXTRACT}" -maxdepth 2 -type d | head -30 >&2
  exit 1
fi

mkdir -p "$(dirname "${TARGET_DIR}")"
if [[ "${FOUND}" == "${TMP_EXTRACT}" ]]; then
  mkdir -p "${TARGET_DIR}"
  shopt -s dotglob
  mv "${TMP_EXTRACT}"/* "${TARGET_DIR}"/
  shopt -u dotglob
else
  mv "${FOUND}" "${TARGET_DIR}"
fi
rm -rf "${TMP_EXTRACT}"

if [[ "${IMAGENETV2_KEEP_TAR:-0}" != "1" ]]; then
  rm -f "${ARCHIVE}"
fi

NUM_IMAGES=$(find "${TARGET_DIR}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l | tr -d ' ')
NUM_CLASSES=$(find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')

echo "[OK] ImageNet-V2 ready: ${TARGET_DIR}"
echo "Classes: ${NUM_CLASSES}, images: ${NUM_IMAGES}"
echo "Test command:"
echo "  python tpt_classification.py ${DATA_ROOT} --test_sets V -a RN50 -b 8 --gpu 0 --tpt --ctx_init a_photo_of_a"
