.PHONY: all run clean fftw

all:
	mkdir -p build && cd build && cmake -S ../ -B ./ && make

run: 
	./build/kcfDetect

clean:
	rm -rf build

fftw:
	./make_fftw.sh
	