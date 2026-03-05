"""Loss ablation: CE / CE+TALS / CE+TAML / Combined"""
import os, json
from topo_guided_training import (
    set_seed, SEED, RESULTS_DIR, get_device, parse_data, DATA_DIR,
    build_topo_distance_matrix, run_topo_experiment
)

set_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
device = get_device()
print(f'Device: {device}', flush=True)

df = parse_data(DATA_DIR)
topo_dist = build_topo_distance_matrix()

configs = [
    ('CE_only', 0.0, 0.0),
    ('TALS_only', 0.1, 0.0),
    ('TAML_only', 0.0, 0.005),
    ('Combined', 0.1, 0.005)
]

results = []
for name, lam_t, lam_m in configs:
    print(f'\n{"="*60}', flush=True)
    print(f'{name}: lambda_topo={lam_t}, lambda_margin={lam_m}', flush=True)
    set_seed(SEED)
    res = run_topo_experiment(
        'resnet18', df, topo_dist, device,
        lambda_topo=lam_t, lambda_margin=lam_m, epochs=20)
    res['config'] = name
    results.append(res)
    print(f'  Test={res["test_acc"]:.4f}', flush=True)

print(f'\n{"="*60}\nSUMMARY\n{"="*60}')
for r in results:
    print(f'{r["config"]:<15} Test={r["test_acc"]:.4f}')

with open(f'{RESULTS_DIR}/loss_ablation.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\n[Saved] results/loss_ablation.json')
