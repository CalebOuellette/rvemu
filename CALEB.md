

Setup env (root `.env`)
```sh
OPENAI_API_BASE_URL=https://api.openai.com
OPENAI_MODEL=gpt-5.3-codex
OPENAI_API_KEY=your_api_key_here
```


Run via CLI
```sh
cargo run -p rvemu-cli -- -k examples/mem_readwrite.bin -d
```


LLM
```sh
cargo run -p rvemu-cli -- -k examples/test.bin --llm
```

LLM with explicit model override
```sh
cargo run -p rvemu-cli -- -k examples/test.bin --llm --model gpt-4.1-mini
```

LLM with instruction limit
```sh
cargo run -p rvemu-cli -- -k examples/mem_readwrite.bin --llm --model gpt-4.1-mini --llm-max-instructions 1
```

LLM mock mode (no API key/network)
```sh
cargo run -p rvemu-cli -- --llm --llm-mock --memory-size 64 --goal-file examples/llm/mock-goal.bin
```
