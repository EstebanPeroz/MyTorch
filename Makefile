BINS := my_torch_analyzer my_torch_generator my_torch_tuner

CONFIG ?= configs/chess.conf
DATASET ?= dataset/full.txt
NETWORK ?= my_torch_network.nn

all: $(BINS)

$(BINS): build

build:
	cargo build --release --frozen --workspace
	cp $(addprefix ./target/release/,$(BINS)) .

# Generates a blank network from CONFIG and trains it on DATASET (with the mirrors), saving it as NETWORK.
train: all
	test -f $(DATASET) || { echo "error: $(DATASET) not found, see docs/dataset.md" >&2; exit 84; }
	rm -rf .train && mkdir .train
	cd .train && ../my_torch_generator ../$(CONFIG) 1
	./my_torch_analyzer --train --mirror --dashboard --save $(NETWORK) .train/$(basename $(notdir $(CONFIG)))_1.nn $(DATASET)
	rm -rf .train

clean:
	cargo clean

fclean: clean
	rm -rf $(BINS) .train

re: fclean all

.PHONY: all build train clean fclean re $(BINS)
