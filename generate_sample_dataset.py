import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

def _make_healthy_image(size=(512,512)):
    """Create a synthetic healthy green leaf image with light texture/noise."""
    base = Image.new('RGB', size, (34, 139, 34))  # base green
    draw = ImageDraw.Draw(base)
    # add subtle veins/noise
    for i in range(60):
        x1 = random.randint(0, size[0])
        y1 = random.randint(0, size[1])
        x2 = x1 + random.randint(-30, 30)
        y2 = y1 + random.randint(-30, 30)
        color = (30 + random.randint(-10,10), 120 + random.randint(-20,20), 30 + random.randint(-10,10))
        draw.line((x1,y1,x2,y2), fill=color, width=random.randint(1,3))
    base = base.filter(ImageFilter.SMOOTH_MORE)
    return base

def _make_diseased_image(size=(512,512), spots=6):
    """Create a synthetic diseased image by adding brown/yellow spots over a green base."""
    img = _make_healthy_image(size)
    draw = ImageDraw.Draw(img)
    for _ in range(spots):
        cx = random.randint(40, size[0]-40)
        cy = random.randint(40, size[1]-40)
        r = random.randint(12, 60)
        # choose spot color (brown/yellow variants)
        if random.random() < 0.6:
            color = (150 + random.randint(-20,20), 80 + random.randint(-20,20), 30 + random.randint(-10,10))  # brown
        else:
            color = (220 + random.randint(-10,10), 180 + random.randint(-30,30), 60 + random.randint(-20,20))  # yellow
        bbox = [cx-r, cy-r, cx+r, cy+r]
        draw.ellipse(bbox, fill=color)
        # add darker center
        inner_r = int(r * random.uniform(0.3, 0.7))
        inner_bbox = [cx-inner_r, cy-inner_r, cx+inner_r, cy+inner_r]
        draw.ellipse(inner_bbox, fill=(max(color[0]-40,0), max(color[1]-40,0), max(color[2]-20,0)))
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2,1.2)))
    return img

def generate_sample_dataset(root='data', classes=None, train_count=40, val_count=12, size=(512,512)):
    """
    Create synthetic dataset under <root>/train/<class>/ and <root>/val/<class>/.
    Default classes: Healthy and Early Blight (adjust as needed).
    """
    if classes is None:
        classes = ['Healthy', 'Early Blight']
    root = Path(root)
    for split, count in [('train', train_count), ('val', val_count)]:
        for cls in classes:
            outdir = root / split / cls
            outdir.mkdir(parents=True, exist_ok=True)
            # generate images
            for i in range(count):
                if cls.lower().startswith('healthy'):
                    img = _make_healthy_image(size=size)
                else:
                    spots = random.randint(3, 10) if split == 'train' else random.randint(2,6)
                    img = _make_diseased_image(size=size, spots=spots)
                fname = f"{cls.replace(' ','_')}_{i+1:03d}.jpg"
                img.save(outdir / fname, quality=85)
    print(f"✓ Generated sample dataset at: {root.resolve()}")
    print("  Classes:", classes)
    print("  Structure: ")
    print("    ", root / 'train' / classes[0], "...")
    print("Run: python train.py --data-dir data")

if __name__ == '__main__':
    # quick-run default
    generate_sample_dataset()
