# ab-k1: min_ms

| lane | row | before min_ms | after min_ms | after / before per round, median [min-max] | rounds |
| :-- | :-- | --: | --: | :-- | --: |
| metal | gen_greedy_p32_n32 | 334.58 | 307.93 | 0.988 [0.886-1.014] | 6 |
| metal | gen_prefill_p32 | 9.68 | 9.54 | 1.008 [0.875-2.199] | 6 |
| metal | gen_temp0_p32_n32 | 316.44 | 306.25 | 1.016 [0.881-1.125] | 6 |
| metal | gen_topk50_p32_n32 | 336.28 | 332.37 | 1.017 [0.855-1.090] | 6 |
| wgpu | gen_greedy_p32_n32 | 792.93 | 778.45 | 0.999 [0.861-1.105] | 6 |
| wgpu | gen_prefill_p32 | 28.70 | 29.20 | 1.020 [0.862-1.088] | 6 |
| wgpu | gen_temp0_p32_n32 | 813.71 | 793.76 | 1.018 [0.874-1.124] | 6 |
| wgpu | gen_topk50_p32_n32 | 849.03 | 809.75 | 0.989 [0.840-1.123] | 6 |

| lane | decode row | before ms / token | after ms / token | after - before ms / token, per round median [min-max] |
| :-- | :-- | --: | --: | :-- |
| metal | gen_greedy_p32_n32 | 10.481 | 9.625 | -0.365 [-1.271-+0.122] |
| metal | gen_temp0_p32_n32 | 9.866 | 9.583 | +0.165 [-1.214-+1.183] |
| metal | gen_topk50_p32_n32 | 10.531 | 10.422 | +0.203 [-1.508-+0.821] |
| wgpu | gen_greedy_p32_n32 | 24.672 | 24.169 | +0.008 [-3.763-+2.946] |
| wgpu | gen_temp0_p32_n32 | 25.323 | 24.663 | +0.478 [-3.413-+3.027] |
| wgpu | gen_topk50_p32_n32 | 26.440 | 25.179 | -0.261 [-4.851-+3.333] |

# ab-k1: median_ms

| lane | row | before median_ms | after median_ms | after / before per round, median [min-max] | rounds |
| :-- | :-- | --: | --: | :-- | --: |
| metal | gen_greedy_p32_n32 | 381.46 | 359.31 | 0.900 [0.750-1.064] | 6 |
| metal | gen_prefill_p32 | 11.39 | 10.08 | 0.987 [0.883-3.435] | 6 |
| metal | gen_temp0_p32_n32 | 346.76 | 334.00 | 1.082 [0.436-1.653] | 6 |
| metal | gen_topk50_p32_n32 | 377.52 | 363.77 | 1.015 [0.732-1.895] | 6 |
| wgpu | gen_greedy_p32_n32 | 872.78 | 914.77 | 1.067 [0.808-1.611] | 6 |
| wgpu | gen_prefill_p32 | 30.40 | 32.31 | 1.070 [0.790-1.224] | 6 |
| wgpu | gen_temp0_p32_n32 | 844.81 | 943.96 | 1.032 [0.855-1.481] | 6 |
| wgpu | gen_topk50_p32_n32 | 894.91 | 856.61 | 0.966 [0.477-1.147] | 6 |

| lane | decode row | before ms / token | after ms / token | after - before ms / token, per round median [min-max] |
| :-- | :-- | --: | --: | :-- |
| metal | gen_greedy_p32_n32 | 11.980 | 11.247 | -1.309 [-4.149-+0.585] |
| metal | gen_temp0_p32_n32 | 10.815 | 10.449 | +0.856 [-12.632-+7.762] |
| metal | gen_topk50_p32_n32 | 11.850 | 11.409 | +0.230 [-3.300-+13.183] |
| wgpu | gen_greedy_p32_n32 | 27.170 | 28.438 | +1.729 [-5.428-+17.498] |
| wgpu | gen_temp0_p32_n32 | 26.268 | 29.403 | +0.842 [-5.758-+12.613] |
| wgpu | gen_topk50_p32_n32 | 27.921 | 26.562 | -1.008 [-29.387-+4.283] |
