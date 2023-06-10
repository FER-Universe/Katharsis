# Install Guide for Winds10/11


1. copy below
```python
def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]

    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    cdef np.ndarray[np.int64_t, ndim=1] order = scores.argsort()[::-1]

    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int64_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=np.int64)
```

2. Uncomment extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]}, in FaceBoxes/utils/build.py line 46

3. run below

```bash
cd FaceBoxes/utils
python3 build.py build_ext --inplace
cd ../..
```

---

[link](https://github.com/cleardusk/3DDFA_V2/issues/12)