NAME := my_torch_analyzer

all: $(NAME)

$(NAME):
	cargo build --release --frozen
	cp ./target/release/$(NAME) .

clean:
	cargo clean

fclean: clean
	rm -f $(NAME)

re: fclean all

.PHONY: all clean fclean re $(NAME)
