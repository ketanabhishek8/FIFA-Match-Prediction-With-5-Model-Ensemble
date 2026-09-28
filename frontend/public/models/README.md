# 3D models

`trophy.glb` is the World Cup trophy shown behind the app (`src/components/TrophyScene.jsx`).
It was generated with [Tripo](https://www.tripo3d.ai/) and then optimized for the web — the
original export was 56 MB and 1.9M triangles, far more than a background element needs:

```bash
npx @gltf-transform/cli optimize trophy-original.glb trophy.glb \
  --compress meshopt --texture-compress webp --texture-size 2048 \
  --simplify true --simplify-ratio 0.25 --simplify-error 0.001
```

The result embeds its textures (base colour, metal/roughness, normal) as WebP and uses
meshopt geometry compression, so the loader needs `MeshoptDecoder` and there are no separate
texture files to serve. Keep the original export out of the repo; re-run the command above on
it if the model is ever replaced.
