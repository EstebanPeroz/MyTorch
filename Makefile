BINS := my_torch_analyzer my_torch_generator

all: $(BINS)

$(BINS): build

build:
	cargo build --release --frozen --workspace
	cp $(addprefix ./target/release/,$(BINS)) .

clean:
	cargo clean

fclean: clean
	rm -f $(BINS)

re: fclean all

.PHONY: all build clean fclean re $(BINS)
