nsys profile -t cuda,nvtx,osrt -o nsys_async python3 test_async_d2h.py
nsys stats ./nsys_async.nsys-rep
nsys-ui ./nsys_async.nsys-rep
