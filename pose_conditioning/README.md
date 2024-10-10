# Installing experiment tools

```
pip install -e .
```

# Diffusion Training

## Training text-to-image
```
cd pose_conditioning/diffusers/examples/text_to_image/
./train.sh
```

## Training text-to-image t2i conditioning adapter
```
cd pose_conditioning/diffusers/examples/t2i_adapter/
./train.sh
```

NOTE: edit/create ```train.sh``` scripts for your tasks accordingly