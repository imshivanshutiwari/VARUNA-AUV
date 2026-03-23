.PHONY: all build test lint clean run train-model generate-model

all: build test

build:
	cargo build --release

test:
	cargo test

lint:
	cargo clippy -- -D warnings
	cargo fmt --check

clean:
	cargo clean
	find . -name "*.pyc" -delete

run:
	cargo run --release --bin varuna-server

generate-model:
	cd model_training && python3 generate_onnx_model.py

train-model:
	cd model_training && python3 train.py

fmt:
	cargo fmt
