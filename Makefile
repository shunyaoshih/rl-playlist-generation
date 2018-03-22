.PHONY: all debug test clean

train:
	python3 main.py

test:
	python3 main.py --mode test

debug:
	python3 main.py --debug 1

debug_test:
	python3 main.py --debug 1 --mode test

clean:
	rm data/train* data/vocab_default.txt
