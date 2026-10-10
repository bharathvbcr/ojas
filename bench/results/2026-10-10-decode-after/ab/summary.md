# min_ms
| lane | row | before min_ms | after min_ms | after / before per round, median [min-max] | rounds |
| :-- | :-- | --: | --: | :-- | --: |
| metal | gen_greedy_p32_n32 | 321.41 | 323.86 | 1.004 [0.961-1.049] | 6 |
| metal | gen_prefill_p32 | 9.71 | 9.31 | 0.949 [0.771-1.410] | 6 |
| metal | gen_topk50_p32_n32 | 333.03 | 343.19 | 1.046 [0.993-1.099] | 6 |
| wgpu | gen_greedy_p32_n32 | 853.42 | 853.54 | 1.013 [0.838-1.046] | 6 |
| wgpu | gen_prefill_p32 | 27.23 | 26.71 | 0.991 [0.353-1.073] | 6 |
| wgpu | gen_topk50_p32_n32 | 878.03 | 887.40 | 1.030 [0.866-1.051] | 6 |

| lane | decode row | before ms / token | after ms / token | after - before ms / token, per round median [min-max] |
| :-- | :-- | --: | --: | :-- |
| metal | gen_greedy_p32_n32 | 10.043 | 10.089 | +0.095 [-0.543-+0.505] |
| metal | gen_topk50_p32_n32 | 10.413 | 10.781 | +0.504 [-0.071-+0.989] |
| wgpu | gen_greedy_p32_n32 | 26.631 | 26.641 | +0.366 [-4.497-+2.826] |
| wgpu | gen_topk50_p32_n32 | 27.416 | 27.733 | +0.738 [-3.769-+2.941] |

# median_ms
| lane | row | before median_ms | after median_ms | after / before per round, median [min-max] | rounds |
| :-- | :-- | --: | --: | :-- | --: |
| metal | gen_greedy_p32_n32 | 342.21 | 569.09 | 1.748 [0.894-4.302] | 6 |
| metal | gen_prefill_p32 | 13.63 | 10.41 | 1.027 [0.111-1.645] | 6 |
| metal | gen_topk50_p32_n32 | 382.74 | 375.09 | 1.027 [0.429-1.076] | 6 |
| wgpu | gen_greedy_p32_n32 | 949.54 | 1302.73 | 1.028 [0.805-4.857] | 6 |
| wgpu | gen_prefill_p32 | 33.10 | 27.73 | 0.855 [0.193-1.259] | 6 |
| wgpu | gen_topk50_p32_n32 | 934.41 | 1027.36 | 1.103 [0.839-4.509] | 6 |

| lane | decode row | before ms / token | after ms / token | after - before ms / token, per round median [min-max] |
| :-- | :-- | --: | --: | :-- |
| metal | gen_greedy_p32_n32 | 10.625 | 17.937 | +8.461 [-1.282-+29.664] |
| metal | gen_topk50_p32_n32 | 12.035 | 11.760 | +0.354 [-27.261-+2.962] |
| wgpu | gen_greedy_p32_n32 | 29.563 | 41.152 | -0.006 [-6.634-+120.236] |
| wgpu | gen_topk50_p32_n32 | 28.702 | 32.139 | +5.060 [-4.529-+127.671] |
