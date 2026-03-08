#!/usr/bin/env python3
"""
Cross-Domain Phone Photo Inference with Per-Factor Analysis.

Photo naming convention (10 photos per knot, 1.jpg - 10.jpg):
    1.jpg  = Tight,  overhead 90°, background A (white paper), indoor light, rope A  [baseline]
    2.jpg  = Tight,  angled  45°, background A, indoor light, rope A  [angle]
    3.jpg  = Tight,  angled  30°, background A, indoor light, rope A  [angle]
    4.jpg  = Loose,  overhead 90°, background A, indoor light, rope A  [looseness]
    5.jpg  = Loose,  angled  45°, background A, indoor light, rope A  [looseness+angle]
    6.jpg  = Tight,  overhead 90°, background B (wood), indoor light, rope A  [background]
    7.jpg  = Tight,  overhead 90°, background C (cloth), indoor light, rope A  [background]
    8.jpg  = Tight,  overhead 90°, background A, natural light, rope A  [lighting]
    9.jpg  = Tight,  overhead 90°, background A, indoor light, rope B  [rope material]
    10.jpg = Tight,  overhead 90°, background B, natural light, rope B  [combined shift]

Usage:
    python phone_inference.py --photo_dir phone_photos --checkpoint checkpoints/swin_t_best.pth --model swin_t
    python phone_inference.py --photo_dir phone_photos --checkpoint checkpoints/resnet18_best.pth --model resnet18

    Outputs: results/phone_crossdomain.json
"""
import os, json, glob, argparse, re
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
NUM_CLASSES = 10

# Photo-index to domain-shift factor mapping (1-indexed photo number)
PHOTO_FACTORS = {
    1:  {'label': 'baseline',       'angle': '90', 'tightness': 'tight', 'background': 'A', 'lighting': 'indoor', 'rope': 'A'},
    2:  {'label': 'angle',          'angle': '45', 'tightness': 'tight', 'background': 'A', 'lighting': 'indoor', 'rope': 'A'},
    3:  {'label': 'angle',          'angle': '30', 'tightness': 'tight', 'background': 'A', 'lighting': 'indoor', 'rope': 'A'},
    4:  {'label': 'looseness',      'angle': '90', 'tightness': 'loose', 'background': 'A', 'lighting': 'indoor', 'rope': 'A'},
    5:  {'label': 'looseness+angle','angle': '45', 'tightness': 'loose', 'background': 'A', 'lighting': 'indoor', 'rope': 'A'},
    6:  {'label': 'background',     'angle': '90', 'tightness': 'tight', 'background': 'B', 'lighting': 'indoor', 'rope': 'A'},
    7:  {'label': 'background',     'angle': '90', 'tightness': 'tight', 'background': 'C', 'lighting': 'indoor', 'rope': 'A'},
    8:  {'label': 'lighting',       'angle': '90', 'tightness': 'tight', 'background': 'A', 'lighting': 'natural','rope': 'A'},
    9:  {'label': 'rope_material',  'angle': '90', 'tightness': 'tight', 'background': 'A', 'lighting': 'indoor', 'rope': 'B'},
    10: {'label': 'combined_shift', 'angle': '90', 'tightness': 'tight', 'background': 'B', 'lighting': 'natural','rope': 'B'},
}

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def load_model(name, checkpoint, device):
    if name == 'resnet18':
        m = models.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, NUM_CLASSES))
    elif name == 'resnet50':
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, NUM_CLASSES))
    elif name == 'swin_t':
        m = models.swin_t(weights=None)
        m.head = nn.Linear(m.head.in_features, NUM_CLASSES)
    elif name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    elif name == 'vit':
        m = models.vit_b_16(weights=None)
        m.heads.head = nn.Linear(m.heads.head.in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {name}")

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    m.load_state_dict(state)
    m = m.to(device)
    m.eval()
    return m

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

def _extract_photo_number(filename):
    """Extract photo number from filename like '1.jpg', '10.jpg', 'ABK_3.jpg'."""
    stem = os.path.splitext(filename)[0]
    # Try pure number first (subfolder mode: 1.jpg, 10.jpg)
    if stem.isdigit():
        return int(stem)
    # Try prefix_number (flat mode: ABK_3.jpg)
    m = re.search(r'_(\d+)$', stem)
    if m:
        return int(m.group(1))
    return None

def discover_photos(photo_dir):
    """Find photos and their labels. Supports subfolder or flat naming."""
    c2i = {c:i for i,c in enumerate(CLASSES)}
    photos = []

    # Try subfolder structure first
    for cls_name in CLASSES:
        cls_dir = os.path.join(photo_dir, cls_name)
        if os.path.isdir(cls_dir):
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.heic', '*.HEIC'):
                for f in sorted(glob.glob(os.path.join(cls_dir, ext))):
                    pnum = _extract_photo_number(os.path.basename(f))
                    factor = PHOTO_FACTORS.get(pnum, {})
                    photos.append({
                        'path': f, 'label': c2i[cls_name], 'class': cls_name,
                        'photo_num': pnum, 'factor': factor
                    })

    # If no subfolders found, try flat naming
    if len(photos) == 0:
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
            for f in sorted(glob.glob(os.path.join(photo_dir, ext))):
                fn = os.path.basename(f)
                cls_name = fn.split('_')[0].upper()
                if cls_name in c2i:
                    pnum = _extract_photo_number(fn)
                    factor = PHOTO_FACTORS.get(pnum, {})
                    photos.append({
                        'path': f, 'label': c2i[cls_name], 'class': cls_name,
                        'photo_num': pnum, 'factor': factor
                    })

    return photos

def main():
    parser = argparse.ArgumentParser(description='Cross-domain phone photo inference')
    parser.add_argument('--photo_dir', type=str, required=True, help='Directory with phone photos')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--model', type=str, default='swin_t',
                        choices=['resnet18','resnet50','efficientnet_b0','vit','swin_t'])
    parser.add_argument('--output', type=str, default='results/phone_crossdomain.json')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load model
    print(f"Loading {args.model} from {args.checkpoint}...")
    model = load_model(args.model, args.checkpoint, device)

    # Discover photos
    photos = discover_photos(args.photo_dir)
    if len(photos) == 0:
        print(f"ERROR: No photos found in {args.photo_dir}")
        print("Expected structure:")
        print("  phone_photos/ABK/photo1.jpg  (subfolder per class)")
        print("  OR phone_photos/ABK_1.jpg    (flat naming)")
        return

    print(f"Found {len(photos)} photos across {len(set(p['class'] for p in photos))} classes")
    for cls in CLASSES:
        n = sum(1 for p in photos if p['class'] == cls)
        if n > 0:
            print(f"  {cls}: {n} photos")

    # Inference
    tf = get_transform()
    preds, labels, details = [], [], []
    with torch.no_grad():
        for p in photos:
            img = Image.open(p['path']).convert('RGB')
            x = tf(img).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = logits.argmax(1).item()
            pred_cls = CLASSES[pred_idx]
            correct = (pred_idx == p['label'])

            preds.append(pred_idx)
            labels.append(p['label'])
            details.append({
                'file': os.path.basename(p['path']),
                'true': p['class'],
                'pred': pred_cls,
                'correct': correct,
                'confidence': float(probs[pred_idx]),
                'photo_num': p.get('photo_num'),
                'factor': p.get('factor', {}).get('label', 'unknown'),
                'top3': [(CLASSES[i], float(probs[i]))
                         for i in probs.argsort(descending=True)[:3]]
            })

            status = "OK" if correct else "WRONG"
            print(f"  [{status}] {os.path.basename(p['path'])}: "
                  f"true={p['class']} pred={pred_cls} conf={probs[pred_idx]:.3f}")

    preds, labels = np.array(preds), np.array(labels)
    acc = (preds == labels).mean()
    report = classification_report(labels, preds, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=range(NUM_CLASSES))

    print(f"\n{'='*50}")
    print(f"  CROSS-DOMAIN RESULTS ({args.model})")
    print(f"{'='*50}")
    print(f"  Overall Accuracy: {acc*100:.1f}% ({int(acc*len(preds))}/{len(preds)})")
    print(f"  Macro F1: {report['macro avg']['f1-score']:.3f}")
    print(classification_report(labels, preds, target_names=CLASSES, zero_division=0))

    # ── Per-factor analysis ──────────────────────────────────
    factor_groups = {
        'baseline':        [1],
        'angle':           [2, 3],
        'looseness':       [4, 5],
        'background':      [6, 7],
        'lighting':        [8],
        'rope_material':   [9],
        'combined_shift':  [10],
    }
    factor_results = {}
    print(f"\n{'='*50}")
    print(f"  PER-FACTOR ANALYSIS")
    print(f"{'='*50}")
    print(f"  {'Factor':<20} {'Acc':>8} {'N':>5}  Description")
    print(f"  {'-'*20} {'-'*8} {'-'*5}  {'-'*30}")

    factor_desc = {
        'baseline':       'Tight, overhead, white paper, indoor, rope A',
        'angle':          'Tight, 30-45° angle, white paper, indoor, rope A',
        'looseness':      'Loose knot, white paper, indoor, rope A',
        'background':     'Tight, overhead, wood/cloth bg, indoor, rope A',
        'lighting':       'Tight, overhead, white paper, natural light, rope A',
        'rope_material':  'Tight, overhead, white paper, indoor, rope B',
        'combined_shift': 'Tight, overhead, wood bg, natural light, rope B',
    }

    for fname, pnums in factor_groups.items():
        f_details = [d for d in details if d.get('photo_num') in pnums]
        if len(f_details) == 0:
            continue
        f_correct = sum(1 for d in f_details if d['correct'])
        f_acc = f_correct / len(f_details)
        factor_results[fname] = {
            'accuracy': f_acc,
            'n_photos': len(f_details),
            'n_correct': f_correct,
            'photo_nums': pnums,
            'description': factor_desc.get(fname, ''),
        }
        print(f"  {fname:<20} {f_acc*100:6.1f}%  {len(f_details):>4}  {factor_desc.get(fname, '')}")

    # Per-factor delta from baseline
    if 'baseline' in factor_results and factor_results['baseline']['n_photos'] > 0:
        base_acc = factor_results['baseline']['accuracy']
        print(f"\n  {'Factor':<20} {'Delta from baseline':>20}")
        print(f"  {'-'*20} {'-'*20}")
        for fname in factor_groups:
            if fname == 'baseline' or fname not in factor_results:
                continue
            delta = factor_results[fname]['accuracy'] - base_acc
            sign = '+' if delta >= 0 else ''
            print(f"  {fname:<20} {sign}{delta*100:6.1f} pp")

    # Save
    result = {
        'model': args.model,
        'checkpoint': args.checkpoint,
        'photo_dir': args.photo_dir,
        'n_photos': len(photos),
        'accuracy': float(acc),
        'macro_f1': float(report['macro avg']['f1-score']),
        'report': report,
        'confusion_matrix': cm.tolist(),
        'factor_analysis': factor_results,
        'details': details
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[Saved] {args.output}")

if __name__ == '__main__':
    main()
