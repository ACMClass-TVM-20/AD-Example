## generate doxygen
doxygen docs/Doxyfile
## format
./tests/lint/git-clang-format.sh -i --rev HEAD~1
./tests/lint/git-black.sh -i --rev HEAD~1
