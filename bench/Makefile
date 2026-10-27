CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra

build/embed: embed.c
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -rf build

.PHONY: clean
