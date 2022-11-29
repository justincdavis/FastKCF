cd extern/fftw-3.3.10
make distclean
sudo ./configure --disable-fortran --enable-threads --with-openmp
sudo make
sudo make install
