nvcc -maxrregcount=255 -arch=sm_89 --cubin -w -Xptxas -v build.cu
python -u "/home/yxdong/llm/tvm-develop/other-repos/AD-Example/matmul_test/test-3.py" 2>&1 | tee -a test-3-output.txt
