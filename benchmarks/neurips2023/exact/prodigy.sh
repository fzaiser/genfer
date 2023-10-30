#!/usr/bin/env bash

set -e

if [ -z ${PRODIGY+x} ]
then
  echo "Set the PRODIGY variable: it must point to the source directory of Prodigy."
  exit 1
fi

poetry run --directory "$PRODIGY" cli --engine ginac main "$@"
