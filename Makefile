build:
	c++ -O3 -Wall -shared -std=c++11 -fPIC $$(uv run python3 -m pybind11 --includes) \
	src/spark/backends/backend_metal.cpp -o src/spark/backends/backend_metal.so
download_data:
	@echo hello world 
clean:
	rm -f ./src/spark/backends/backend_metal.so

main:
	uv run main.py

install:
	uv pip install spark -e .

.PHONY: build main install clean