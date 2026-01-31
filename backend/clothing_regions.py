def get_prompts(kind, h, w):
    presets = {
        "pants": {
            "points": [[w // 2, int(h * 0.75)], [w // 2, int(h * 0.35)]],
            "labels": [1, 0],
        },
        "shirt": {
            "points": [[w // 2, int(h * 0.38)], [w // 2, int(h * 0.75)]],
            "labels": [1, 0],
        },
        "dress": {
            "points": [[w // 2, int(h * 0.45)], [w // 2, int(h * 0.15)]],
            "labels": [1, 0],
        },
    }
    return presets.get(kind)
