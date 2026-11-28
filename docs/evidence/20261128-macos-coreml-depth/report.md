# macOS Core ML Depth Matrix

- Platform: `macOS-15.2-arm64-arm-64bit`
- Machine: `arm64`

## Offline Depth and Half-SBS

| Model | Resolution | Status | Units | Setup ms | Depth p50/p90 ms | Model p50 ms | Finite |
|---|---:|---|---|---:|---:|---:|---|
| DepthPro2-B | 320 | ok | gpu | 3.6 | 59.8/62.0 | 54.9 | True |
| DepthPro2-B | 512 | ok | gpu | 3.7 | 129.6/132.3 | 124.0 | True |
| DepthPro2-H | 320 | ok | gpu | 5.0 | 377.0/379.0 | 369.2 | True |
| DepthPro2-H | 512 | ok | gpu | 4.4 | 938.0/938.0 | 932.6 | True |
| DepthPro2-L | 320 | ok | gpu | 4.5 | 179.7/180.4 | 173.3 | True |
| DepthPro2-L | 512 | ok | gpu | 4.6 | 422.9/424.9 | 415.9 | True |
| DepthPro2-S | 320 | ok | gpu | 4.1 | 22.9/23.1 | 18.6 | True |
| DepthPro2-S | 512 | ok | gpu | 3.5 | 46.9/49.6 | 42.0 | True |
| Distill-Any-Depth-Base | 518 | ok | - | 4.0 | 68.4/69.3 | 63.3 | True |
| Distill-Any-Depth-Small | 336 | ok | - | 4.4 | 13.7/15.5 | 9.0 | True |
| FlashDepth-Full | 336 | ok | gpu | 3.9 | 162.9/164.6 | 157.0 | True |
| FlashDepth-Full | 518 | ok | gpu | 3.8 | 377.7/379.3 | 370.4 | True |
| FlashDepth-L | 336 | ok | gpu | 1.9 | 172.1/174.1 | 165.9 | True |
| FlashDepth-L | 518 | ok | gpu | 4.2 | 401.6/402.3 | 395.2 | True |
| FlashDepth-S | 336 | ok | gpu | 4.9 | 29.7/33.2 | 24.7 | True |
| FlashDepth-S | 518 | ok | gpu | 2.4 | 52.5/52.6 | 47.2 | True |
| MoGe3-G | 336 | ok | gpu | 7.2 | 6403.0/6554.1 | 6376.6 | True |
| MoGe3-G | 518 | ok | gpu | 4.4 | 658531.3/658531.3 | 81173.7 | True |
| MoGe3-L | 336 | ok | gpu | 10.0 | 2002.3/2008.1 | 1979.1 | True |
| MoGe3-L | 518 | ok | gpu | 5.4 | 3920.6/3924.3 | 3898.0 | True |

Latency ratios below compare each candidate against the Distill model at the same tier:

| Candidate | Resolution | Candidate p50 ms | Baseline | Baseline p50 ms | Ratio |
|---|---:|---:|---|---:|---:|
| DepthPro2-B | 320 | 59.8 | Distill-Any-Depth-Small | 13.7 | 4.36x |
| DepthPro2-B | 512 | 129.6 | Distill-Any-Depth-Base | 68.4 | 1.89x |
| DepthPro2-H | 320 | 377.0 | Distill-Any-Depth-Small | 13.7 | 27.48x |
| DepthPro2-H | 512 | 938.0 | Distill-Any-Depth-Base | 68.4 | 13.71x |
| DepthPro2-L | 320 | 179.7 | Distill-Any-Depth-Small | 13.7 | 13.10x |
| DepthPro2-L | 512 | 422.9 | Distill-Any-Depth-Base | 68.4 | 6.18x |
| DepthPro2-S | 320 | 22.9 | Distill-Any-Depth-Small | 13.7 | 1.67x |
| DepthPro2-S | 512 | 46.9 | Distill-Any-Depth-Base | 68.4 | 0.69x |
| FlashDepth-Full | 336 | 162.9 | Distill-Any-Depth-Small | 13.7 | 11.88x |
| FlashDepth-Full | 518 | 377.7 | Distill-Any-Depth-Base | 68.4 | 5.52x |
| FlashDepth-L | 336 | 172.1 | Distill-Any-Depth-Small | 13.7 | 12.54x |
| FlashDepth-L | 518 | 401.6 | Distill-Any-Depth-Base | 68.4 | 5.87x |
| FlashDepth-S | 336 | 29.7 | Distill-Any-Depth-Small | 13.7 | 2.17x |
| FlashDepth-S | 518 | 52.5 | Distill-Any-Depth-Base | 68.4 | 0.77x |
| MoGe3-G | 336 | 6403.0 | Distill-Any-Depth-Small | 13.7 | 466.77x |
| MoGe3-G | 518 | 658531.3 | Distill-Any-Depth-Base | 68.4 | 9626.10x |
| MoGe3-L | 336 | 2002.3 | Distill-Any-Depth-Small | 13.7 | 145.96x |
| MoGe3-L | 518 | 3920.6 | Distill-Any-Depth-Base | 68.4 | 57.31x |

## Local Viewer

| Model | Backend | Zero Copy | Present FPS | Content FPS | Latency p50/p95 ms | Depth p95 ms | Runtime p95 ms |
|---|---|---|---:|---:|---:|---:|---:|
| DepthPro2-B | coreml | yes | 55.3 | 7.8 | 145.0/188.1 | 137.7 | 138.1 |
| DepthPro2-H | coreml | yes | 52.8 | 1.0 | 1000.2/1064.5 | 1018.2 | 1019.4 |
| DepthPro2-L | coreml | yes | 52.9 | 2.4 | 443.5/485.8 | 429.3 | 430.8 |
| DepthPro2-S | coreml | yes | 64.7 | 20.9 | 57.8/84.4 | 45.4 | 46.1 |
| FlashDepth-Full | coreml | yes | 51.7 | 2.6 | 398.1/445.7 | 384.0 | 385.2 |
| FlashDepth-L | coreml | yes | 51.7 | 2.5 | 421.3/467.9 | 405.6 | 406.9 |
| FlashDepth-S | coreml | yes | 51.6 | 15.4 | 65.1/90.6 | 49.7 | 50.6 |
| MoGe3-G | coreml | yes | 50.7 | 0.2 | 7482.0/7981.6 | 7966.2 | 7969.6 |
| MoGe3-L | coreml | yes | 51.5 | 0.5 | 2027.7/2272.2 | 2243.5 | 2251.3 |

## Contact Sheets

- [DepthPro2-B_depth_half_sbs.jpg](contact_sheets/DepthPro2-B_depth_half_sbs.jpg)
- [DepthPro2-H_depth_half_sbs.jpg](contact_sheets/DepthPro2-H_depth_half_sbs.jpg)
- [DepthPro2-L_depth_half_sbs.jpg](contact_sheets/DepthPro2-L_depth_half_sbs.jpg)
- [DepthPro2-S_depth_half_sbs.jpg](contact_sheets/DepthPro2-S_depth_half_sbs.jpg)
- [FlashDepth-Full_depth_half_sbs.jpg](contact_sheets/FlashDepth-Full_depth_half_sbs.jpg)
- [FlashDepth-L_depth_half_sbs.jpg](contact_sheets/FlashDepth-L_depth_half_sbs.jpg)
- [FlashDepth-S_depth_half_sbs.jpg](contact_sheets/FlashDepth-S_depth_half_sbs.jpg)
- [MoGe3-G_depth_half_sbs.jpg](contact_sheets/MoGe3-G_depth_half_sbs.jpg)
- [MoGe3-L_depth_half_sbs.jpg](contact_sheets/MoGe3-L_depth_half_sbs.jpg)

## Notes

- Local Viewer ran the Metal zero-copy path; every row reports Core ML on GPU with no fallback.
- MoGe-3 is much slower than the other families and is best tested at 336 for live use.
- A first MoGe3-G@518 conversion includes one-time graph compilation and is not a steady-state frame latency.
