#!/usr/bin/env bash

# Shared configuration and helpers for the DGX Spark job scripts.

set -euo pipefail

SPARK_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SPARK_PROJECT_ROOT="$(cd -- "${SPARK_SCRIPT_DIR}/.." && pwd)"

if [[ -f "${SPARK_PROJECT_ROOT}/.spark.env" ]]; then
    # shellcheck disable=SC1091
    source "${SPARK_PROJECT_ROOT}/.spark.env"
fi

SPARK_HOST="${SPARK_HOST:-spark-3946}"
SPARK_PROJECT_DIR="${SPARK_PROJECT_DIR:-/home/liuyang/vlx_seek_1_5_3b}"
SPARK_REMOTE_JOBS_DIR="${SPARK_REMOTE_JOBS_DIR:-${SPARK_PROJECT_DIR}/.spark-jobs}"
SPARK_LOCAL_RESULTS_DIR="${SPARK_LOCAL_RESULTS_DIR:-${SPARK_PROJECT_ROOT}/.spark-results}"
SPARK_SSH_BIN="${SPARK_SSH_BIN:-ssh}"
SPARK_SCP_BIN="${SPARK_SCP_BIN:-scp}"
SPARK_RSYNC_BIN="${SPARK_RSYNC_BIN:-rsync}"

spark_die() {
    echo "error: $*" >&2
    exit 1
}

spark_quote() {
    printf '%q' "$1"
}

spark_validate_job_id() {
    [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
        spark_die "invalid job id: $1"
}

spark_resolve_job_id() {
    local requested="${1:-latest}"
    if [[ "${requested}" != "latest" ]]; then
        spark_validate_job_id "${requested}"
        printf '%s\n' "${requested}"
        return
    fi

    local remote_jobs_q
    remote_jobs_q="$(spark_quote "${SPARK_REMOTE_JOBS_DIR}")"
    "${SPARK_SSH_BIN}" "${SPARK_HOST}" \
        "test -s ${remote_jobs_q}/.latest && cat ${remote_jobs_q}/.latest" || \
        spark_die "no submitted Spark job was found"
}
