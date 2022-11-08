.PHONY: all run clean

all:
	mkdir -p build && cd build && cmake -S ../ -B ./ && make

run: 
	./build/kcfDetect

clean:
	rm -rf build
	