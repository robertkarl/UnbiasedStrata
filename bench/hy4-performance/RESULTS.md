# Stock Strata measurements

Each SQ-128 row used a fresh cache, a 4,750-token input and a 131,072-token context. Speeds are native tokens/s.

| GPU placement | Output tokens | Prompt | Decode |
| --- | ---: | ---: | ---: |
| 4090 | 2048 | 65.54 | 9.74 |
| a6000 | 2048 | 88.77 | 12.52 |
| dual | 2048 | 144.43 | 19.87 |
