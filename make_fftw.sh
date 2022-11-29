cd extern/fftw-3.3.10
make distclean
sudo ./configure --enable-openmp --enable-avx --enable-generic-simd128 --enable-generic-simd256
sudo make
sudo make install
