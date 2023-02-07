#!/bin/bash
echo -e "---------------------black---------------------\n"
tests/lint/git-black.sh -i --rev HEAD~10
echo -e "\n\n---------------------clang-format---------------------\n"
tests/lint/git-clang-format.sh -i --rev HEAD~10
echo -e "\n\n---------------------pylint---------------------\n"
python3 -m pylint python/tvm/relax/training --rcfile=tests/lint/pylintrc
echo -e "\n\n---------------------mypy---------------------\n"
mypy --check-untyped-defs python/tvm/relax/
