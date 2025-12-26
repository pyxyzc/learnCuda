# nsys profile -t cuda,nvtx,osrt -o nsys_async python3 test_async_d2h.py
# nsys stats ./nsys_async.nsys-rep
# nsys-ui ./nsys_async.nsys-rep
rm -rf profile_esa.* ; CUDA_VISIBLE_DEVICES=2 nsys profile -t cuda,nvtx,osrt -o profile_esa python3 test_esa_interface.py
