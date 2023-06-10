# Install Guide for Wins10/11


1. install conda gcc

```bash
conda install -c conda-forge m2w64-gcc
```

2. go to and run
```bash
cd utils/asset
gcc -shared -Wall -O3 render.c -o render.so -fPIC
cd ../..
```