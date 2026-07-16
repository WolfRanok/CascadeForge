import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cascadeforge.config import load_config
from cascadeforge.organize import split_grid
from cascadeforge.preprocess import crop_size, iou, select_candidates
from cascadeforge.select import validate_response


def test_crop_size_prefers_supported_ratio():
    assert crop_size(1600, 900) == (1600, 900, "16:9")


def test_iou_and_candidate_filter_remove_nested_masks():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 1
    nested = mask.copy()
    separate = np.zeros_like(mask)
    separate[60:90, 60:90] = 1
    annotations = [
        {"id": 1, "predicted_iou": 0.99, "stability_score": 0.99},
        {"id": 2, "predicted_iou": 0.99, "stability_score": 0.99},
        {"id": 3, "predicted_iou": 0.99, "stability_score": 0.99},
        {"id": 4, "predicted_iou": 0.99, "stability_score": 0.99},
    ]
    candidates = select_candidates(annotations, [mask, nested, separate, separate], 100, 100)
    assert iou(mask, nested) == 1.0
    assert len(candidates) == 2


def test_validate_response_orders_rounds_by_area(tmp_path):
    masks_path = tmp_path / "masks.npz"
    np.savez(
        masks_path,
        **{
            "1": np.pad(np.ones((2, 2), dtype=np.uint8), ((0, 6), (0, 6))),
            "2": np.pad(np.ones((2, 2), dtype=np.uint8), ((0, 6), (6, 0))),
            "3": np.pad(np.ones((2, 2), dtype=np.uint8), ((6, 0), (0, 6))),
            "4": np.pad(np.ones((2, 2), dtype=np.uint8), ((6, 0), (6, 0))),
        },
    )
    masks = np.load(masks_path)
    data = {
        "selected_ids": [1, 2, 3, 4],
        **{f"ROUND_{i}": {"short": str(i), "long": f"修改目标 {i} 的颜色"} for i in range(1, 5)},
    }
    selected, rounds = validate_response(
        data,
        masks,
        [{"candidate_id": 1, "area": 1}, {"candidate_id": 2, "area": 40}, {"candidate_id": 3, "area": 20}, {"candidate_id": 4, "area": 8}],
    )
    assert selected == ["2", "3", "4", "1"]
    assert set(rounds) == {"ROUND_1", "ROUND_2", "ROUND_3", "ROUND_4"}


def test_validate_response_rejects_candidate_id_in_instruction(tmp_path):
    masks_path = tmp_path / "masks.npz"
    masks = {str(index): np.pad(np.ones((2, 2), dtype=np.uint8), ((0 if index < 3 else 6, 6 if index < 3 else 0), (0 if index % 2 else 6, 6 if index % 2 else 0))) for index in range(1, 5)}
    np.savez(masks_path, **masks)
    response = {
        "selected_ids": [1, 2, 3, 4],
        **{f"ROUND_{i}": {"short": "修改", "long": f"将编号{i}目标改色"} for i in range(1, 5)},
    }
    with pytest.raises(ValueError, match="编号"):
        validate_response(response, np.load(masks_path), [{"candidate_id": i, "area": 4} for i in range(1, 5)])


def test_organize_splits_grid(tmp_path):
    edited = tmp_path / "abc_gpt_edited.jpg"
    source = tmp_path / "abc.jpg"
    image = Image.new("RGB", (8, 8), "white")
    image.save(edited)
    image.save(source)
    ok, digest, message = split_grid(edited, tmp_path / "out", source)
    assert ok and digest == "abc"
    assert (tmp_path / "out" / "abc" / "ROUND_4.jpg").exists()


def test_config_environment_overrides_local(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"moliapi": {"api_key": "local", "model": "local-model"}}), encoding="utf-8")
    monkeypatch.setenv("GPT_API_KEY", "environment")
    monkeypatch.setenv("GPT_MODEL", "environment-model")
    config = load_config(config_path)
    assert config.vision.api_key == "environment"
    assert config.vision.model == "environment-model"
