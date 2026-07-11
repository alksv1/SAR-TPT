#!/usr/bin/env bash
set -euo pipefail

# Download Oxford-IIIT Pets for this repo.
# This project expects the extracted folder to be:
#   ${DATA_ROOT}/OxfordPets/images/*.jpg
# and the CoOp split json to be:
#   data/data_splits/split_zhou_OxfordPets.json
#
# Usage:
#   bash scripts/download_pets.sh /path/to/data/root
#   bash scripts/download_pets.sh /path/to/data/root --images-url https://your-mirror/images.tar.gz
#   bash scripts/download_pets.sh /path/to/data/root --split-url https://your-mirror/split_zhou_OxfordPets.json
#
# Environment variables:
#   PETS_IMAGES_URL       Override images.tar.gz download URL.
#   PETS_ANNOTATIONS_URL  Override annotations.tar.gz download URL.
#   PETS_SPLIT_URL        Override CoOp split json download URL.
#   PETS_KEEP_TAR         Set to 1 to keep downloaded .tar.gz files.

DATA_ROOT="${1:-}"
if [[ -z "${DATA_ROOT}" || "${DATA_ROOT}" == "-h" || "${DATA_ROOT}" == "--help" ]]; then
  sed -n '1,20p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi
shift || true

CUSTOM_IMAGES_URL="${PETS_IMAGES_URL:-}"
CUSTOM_ANNOTATIONS_URL="${PETS_ANNOTATIONS_URL:-}"
CUSTOM_SPLIT_URL="${PETS_SPLIT_URL:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --images-url)
      CUSTOM_IMAGES_URL="${2:-}"
      shift 2
      ;;
    --annotations-url)
      CUSTOM_ANNOTATIONS_URL="${2:-}"
      shift 2
      ;;
    --split-url)
      CUSTOM_SPLIT_URL="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

TARGET_NAME="OxfordPets"
TARGET_DIR="${DATA_ROOT}/${TARGET_NAME}"
IMAGES_DIR="${TARGET_DIR}/images"
ANNOTATIONS_DIR="${TARGET_DIR}/annotations"
IMAGES_ARCHIVE="${DATA_ROOT}/oxford-pets-images.tar.gz"
ANNOTATIONS_ARCHIVE="${DATA_ROOT}/oxford-pets-annotations.tar.gz"
TMP_EXTRACT="${DATA_ROOT}/.oxfordpets_extract_tmp"
SPLIT_DIR="data/data_splits"
SPLIT_FILE="${SPLIT_DIR}/split_zhou_OxfordPets.json"

IMAGES_URLS=()
if [[ -n "${CUSTOM_IMAGES_URL}" ]]; then
  IMAGES_URLS+=("${CUSTOM_IMAGES_URL}")
fi
IMAGES_URLS+=(
  "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
)

ANNOTATIONS_URLS=()
if [[ -n "${CUSTOM_ANNOTATIONS_URL}" ]]; then
  ANNOTATIONS_URLS+=("${CUSTOM_ANNOTATIONS_URL}")
fi
ANNOTATIONS_URLS+=(
  "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
)

SPLIT_URLS=()
if [[ -n "${CUSTOM_SPLIT_URL}" ]]; then
  SPLIT_URLS+=("${CUSTOM_SPLIT_URL}")
fi
SPLIT_URLS+=(
  "https://raw.githubusercontent.com/KaiyangZhou/CoOp/main/splits/oxford_pets/split_zhou_OxfordPets.json"
  "https://github.com/KaiyangZhou/CoOp/raw/main/splits/oxford_pets/split_zhou_OxfordPets.json"
)

mkdir -p "${DATA_ROOT}" "${TARGET_DIR}" "${SPLIT_DIR}"

_download_one() {
  local output="$1"
  shift
  local urls=("$@")

  if [[ -s "${output}" ]]; then
    echo "[OK] File already exists: ${output}"
    return 0
  fi

  echo "Downloading to: ${output}"
  local success=0
  for url in "${urls[@]}"; do
    echo "Trying: ${url}"
    if command -v aria2c >/dev/null 2>&1; then
      if aria2c -x 8 -s 8 -k 1M --continue=true --auto-file-renaming=false \
          --dir "$(dirname "${output}")" --out "$(basename "${output}")" "${url}"; then
        success=1
        break
      fi
    elif command -v wget >/dev/null 2>&1; then
      if wget -c -O "${output}" "${url}"; then
        success=1
        break
      fi
    elif command -v curl >/dev/null 2>&1; then
      if curl -L --fail --retry 3 -C - -o "${output}" "${url}"; then
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
    echo "All download URLs failed for ${output}. Try a custom URL." >&2
    exit 1
  fi
}

_extract_tar_gz() {
  local archive="$1"
  local tmp_dir="$2"
  rm -rf "${tmp_dir}"
  mkdir -p "${tmp_dir}"
  echo "Extracting: ${archive}"
  if gzip -t "${archive}" >/dev/null 2>&1; then
    tar -xzf "${archive}" -C "${tmp_dir}"
  else
    echo "Archive is not gzip-compressed; trying plain tar extraction..."
    if ! tar -xf "${archive}" -C "${tmp_dir}"; then
      echo "Extraction failed. File type:" >&2
      file "${archive}" >&2 || true
      exit 1
    fi
  fi
}

if [[ ! -d "${IMAGES_DIR}" ]]; then
  _download_one "${IMAGES_ARCHIVE}" "${IMAGES_URLS[@]}"
  _extract_tar_gz "${IMAGES_ARCHIVE}" "${TMP_EXTRACT}"
  if [[ -d "${TMP_EXTRACT}/images" ]]; then
    rm -rf "${IMAGES_DIR}"
    mv "${TMP_EXTRACT}/images" "${IMAGES_DIR}"
  else
    echo "Could not find extracted images folder under ${TMP_EXTRACT}" >&2
    find "${TMP_EXTRACT}" -maxdepth 2 -type d | head -30 >&2
    exit 1
  fi
  rm -rf "${TMP_EXTRACT}"
else
  echo "[OK] Images already exist: ${IMAGES_DIR}"
fi

if [[ ! -d "${ANNOTATIONS_DIR}" ]]; then
  _download_one "${ANNOTATIONS_ARCHIVE}" "${ANNOTATIONS_URLS[@]}"
  _extract_tar_gz "${ANNOTATIONS_ARCHIVE}" "${TMP_EXTRACT}"
  if [[ -d "${TMP_EXTRACT}/annotations" ]]; then
    rm -rf "${ANNOTATIONS_DIR}"
    mv "${TMP_EXTRACT}/annotations" "${ANNOTATIONS_DIR}"
  else
    echo "Could not find extracted annotations folder under ${TMP_EXTRACT}" >&2
    find "${TMP_EXTRACT}" -maxdepth 2 -type d | head -30 >&2
    exit 1
  fi
  rm -rf "${TMP_EXTRACT}"
else
  echo "[OK] Annotations already exist: ${ANNOTATIONS_DIR}"
fi

_download_one "${SPLIT_FILE}" "${SPLIT_URLS[@]}"

if [[ "${PETS_KEEP_TAR:-0}" != "1" ]]; then
  rm -f "${IMAGES_ARCHIVE}" "${ANNOTATIONS_ARCHIVE}"
fi

NUM_IMAGES=$(find "${IMAGES_DIR}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l | tr -d ' ')
NUM_ANNOS=$(find "${ANNOTATIONS_DIR}" -type f | wc -l | tr -d ' ')

if [[ ! -s "${SPLIT_FILE}" ]]; then
  echo "Split file is missing or empty: ${SPLIT_FILE}" >&2
  exit 1
fi

if [[ "${NUM_IMAGES}" -eq 0 ]]; then
  echo "No images found under ${IMAGES_DIR}" >&2
  exit 1
fi

echo "[OK] OxfordPets ready: ${TARGET_DIR}"
echo "Images: ${NUM_IMAGES}, annotation files: ${NUM_ANNOS}"
echo "Split: ${SPLIT_FILE}"
echo "Test command:"
echo "  python tpt_classification.py ${DATA_ROOT} --test_sets Pets -a ViT-B/16 -b 8 --gpu 0 --tpt --ctx_init a_photo_of_a"
