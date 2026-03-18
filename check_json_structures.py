#!/usr/bin/env python3
"""Quick check of JSON structures."""
import json, os
RESULTS_DIR = 'results'

print("=== CUB-200 ===")
with open(f'{RESULTS_DIR}/cub200_taca_results.json') as f:
    d = json.load(f)
print(f"Type: {type(d).__name__}")
if isinstance(d, dict):
    for k, v in d.items():
        print(f"  {k}: {type(v).__name__}", end="")
        if isinstance(v, dict):
            print(f" -> keys: {list(v.keys())[:10]}")
        elif isinstance(v, (int, float, str)):
            print(f" = {v}")
        elif isinstance(v, list):
            print(f" len={len(v)}")
        else:
            print()

print("\n=== Aircraft ===")
with open(f'{RESULTS_DIR}/aircraft_taca_results.json') as f:
    d2 = json.load(f)
print(f"Type: {type(d2).__name__}")
if isinstance(d2, list):
    print(f"  len={len(d2)}")
    for i, item in enumerate(d2[:3]):
        if isinstance(item, dict):
            print(f"  [{i}]: keys={list(item.keys())[:10]}")
            for k2, v2 in list(item.items())[:5]:
                if isinstance(v2, (int, float, str)):
                    print(f"       {k2} = {v2}")
elif isinstance(d2, dict):
    for k, v in list(d2.items())[:10]:
        print(f"  {k}: {type(v).__name__}", end="")
        if isinstance(v, (int, float, str)):
            print(f" = {v}")
        else:
            print()

print("\n=== Embedding V2 ===")
with open(f'{RESULTS_DIR}/embedding_alignment_v2.json') as f:
    d3 = json.load(f)
for k in list(d3.keys())[:3]:
    v = d3[k]
    if isinstance(v, dict):
        print(f"  {k}: {list(v.keys())[:8]}")
        for k2 in list(v.keys())[:3]:
            v2 = v[k2]
            if isinstance(v2, (int, float, str)):
                print(f"    {k2} = {v2}")
            elif isinstance(v2, list):
                print(f"    {k2}: list[{len(v2)}]")
