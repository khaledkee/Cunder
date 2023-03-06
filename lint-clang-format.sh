#!/usr/bin/env bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

git ls-files *.c *.cpp *.h :!:external | xargs clang-format -style=file -i
if git diff --exit-code; then
	echo -e "[${GREEN}OK${NC}]: lint-clang-format.sh"
else
	echo -e "[${RED}FAIL${NC}]: lint-clang-format.sh"
	exit -1
fi