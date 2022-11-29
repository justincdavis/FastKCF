cd extern/fftw-3.3.10
make distclean
./configure --enable-openmp --enable-avx --enable-generic-simd128 --enable-generic-simd256 --enable-sse --enable-sse2
make
sudo make install
