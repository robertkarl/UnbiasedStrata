# Stock Strata measurements

Each SQ-128 row used a fresh cache, a 4,750-token input and a 131,072-token context. Speeds are native tokens/s.

| GPU placement | Output tokens | Prompt | Decode |
| --- | ---: | ---: | ---: |
| 4090 | 2048 | 65.54 | 9.74 |
| a6000 | 2048 | 88.77 | 12.52 |
| dual | 2048 | 144.43 | 19.87 |

## SQ-64 at 4K context

Dual-V100 reference 1,068 prompt and 41.8 decode tokens/s. The 80% floors are 854.4 and 33.44. Automatic chunks are denoted by 0.

| Stage | GPU order | Chunk | Temperature | Output tokens per trial | Prompt | Decode |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| screen | 4090-a6000 | 0 | 0.9 | 256 | 2192.72 | 90.94 |
