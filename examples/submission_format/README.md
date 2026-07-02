# Submission format example

```
preds/
  bear/
    orbit_left/pred.mp4      # >= 10 frames
    zoom_in/pred.mp4
  camel/
    orbit_left/pred.mp4
cameras/
  bear/
    orbit_left.npz           # np.savez(cam_c2w=<(T,4,4) float32 c2w>)
    zoom_in.npz
  camel/
    orbit_left.npz
sources/                     # optional (CLIP-V; DAVIS flat-FVD reference)
  bear/orbit_left/source.mp4
gt/                          # Syn4D track only (paired target-view GT)
  bear/orbit_left/gt.mp4
prompts.json                 # optional: {"bear": "a bear walking", "camel": "..."}
pairs.csv                    # optional: header `video,trajectory`, one row per pair
```

`cam_c2w` are camera-to-world 4x4 matrices, frame 0 ~ identity, OpenCV convention
(+Z forward, -Y up) — the requested target trajectory the model was asked to follow.
Metrics are alignment-invariant (Sim(3)), so world frame/scale need not match.
```
```
