#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${1:-$ROOT_DIR/.env}"
VOLUME_NAME="${POSTGRES_VOLUME_NAME:-bevp_postgres_data}"
DB_NAME="${POSTGRES_DB:-bevp_db}"
DB_USER="${POSTGRES_USER:-bevp_user}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

PASSWORD="$(awk -F= '/^POSTGRES_PASSWORD=/{sub(/\r$/, "", $0); print substr($0, index($0, "=") + 1)}' "$ENV_FILE" | tail -1)"
PASSWORD="${PASSWORD%\"}"
PASSWORD="${PASSWORD#\"}"

if [[ -z "$PASSWORD" ]]; then
  echo "POSTGRES_PASSWORD is not set in $ENV_FILE" >&2
  exit 1
fi

ESCAPED_PASSWORD="${PASSWORD//\'/\'\'}"
SQL_FILE="$(mktemp)"
trap 'rm -f "$SQL_FILE"' EXIT

inspect_pg() {
  local sql="$1"
  docker run --rm -i \
    -v "$VOLUME_NAME:/var/lib/postgresql/data" \
    --user postgres \
    postgres:16-alpine \
    postgres --single -D /var/lib/postgresql/data template1 <<<"$sql" 2>/dev/null || true
}

role_exists="$(inspect_pg "SELECT rolname FROM pg_authid;" | grep -F "rolname = \"$DB_USER\"" || true)"
db_exists="$(inspect_pg "SELECT datname FROM pg_database;" | grep -F "datname = \"$DB_NAME\"" || true)"

if [[ -z "$role_exists" ]]; then
  printf "CREATE ROLE %s LOGIN CREATEDB PASSWORD '%s';\n" "$DB_USER" "$ESCAPED_PASSWORD" >>"$SQL_FILE"
fi

if [[ -z "$db_exists" ]]; then
  printf "CREATE DATABASE %s OWNER %s;\n" "$DB_NAME" "$DB_USER" >>"$SQL_FILE"
fi

if [[ ! -s "$SQL_FILE" ]]; then
  echo "Postgres role/database already present for $DB_USER/$DB_NAME."
  exit 0
fi

docker run --rm -i \
  -v "$VOLUME_NAME:/var/lib/postgresql/data" \
  --user postgres \
  postgres:16-alpine \
  postgres --single -D /var/lib/postgresql/data template1 <"$SQL_FILE"

echo "Postgres role/database repair complete for $DB_USER/$DB_NAME."
