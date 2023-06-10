# Install Guide for Wins10/11


1. change `bfm` source code in line 34

- Previous

```python
self.keypoints = bfm.get('keypoints').astype(np.long)  # fix bug
```

- After

```python
self.keypoints = bfm.get('keypoints').astype(np.int64)
```


2. Add below in all `demo` python files

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
```