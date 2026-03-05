from PIL import Image
import os

figures = [
    'difficulty_tiers',
    'model_comparison',
    'f1_heatmap',
    'convergence_all',
    'confusion_best_worst'
]

for fig in figures:
    png = f'results/figures/{fig}.png'
    pdf = f'results/figures/{fig}.pdf'
    if os.path.exists(png):
        img = Image.open(png).convert('RGB')
        img.save(pdf, 'PDF', resolution=300.0)
        print(f'✓ {fig}.pdf')
    else:
        print(f'✗ {fig}.png not found')
