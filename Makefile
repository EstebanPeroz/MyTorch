BINS := my_torch_analyzer my_torch_generator

all: $(BINS)

$(BINS):
	cargo build --release --frozen --workspace
	cp ./target/release/my_torch_analyzer .
	cp ./target/release/my_torch_generator .

clean:
	cargo clean

fclean: clean
	rm -f $(BINS)

re: fclean all

.PHONY: all clean fclean re $(BINS)
