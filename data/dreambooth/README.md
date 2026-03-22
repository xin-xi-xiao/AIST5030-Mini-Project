# DreamBooth Data Layout

Place your subject images in the following folders:

- `data/dreambooth/dog/`
- `data/dreambooth/backpack/`
- `data/dreambooth/cat/`

Each folder should contain 4-6 clean images (jpg/png/webp) of one subject.

Example command for dog experiments:

```bash
bash scripts/run_all.sh dog data/dreambooth/dog "a photo of sks dog" 800
```
