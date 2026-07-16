import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cascadeforge.config import load_config
from cascadeforge.organize import split_grid
from cascadeforge.preprocess import crop_size, iou, select_candidates
from cascadeforge.select import (
    PROMPT,
    SELECTION_MODE,
    is_current_selection,
    make_outputs,
    process_one as select_one,
    validate_response,
)


def test_crop_size_prefers_supported_ratio():
    assert crop_size(1600, 900) == (1600, 900, "16:9")


def test_selection_prompt_preserves_short_and_long_length_rules():
    assert "ROUND_1 至 ROUND_4" in PROMPT
    assert "short 均不超过 10 个汉字" in PROMPT
    assert "long 均为 15–30 个汉字" in PROMPT


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
        },
    )
    masks = np.load(masks_path)
    data = {
        "selected_ids": [1, 2, 3],
        **{f"ROUND_{i}": {"short": str(i), "long": f"修改目标 {i} 的颜色"} for i in range(1, 5)},
    }
    selected, rounds = validate_response(
        data,
        masks,
        [{"candidate_id": 1, "area": 1}, {"candidate_id": 2, "area": 40}, {"candidate_id": 3, "area": 20}],
    )
    assert selected == ["2", "3", "1"]
    assert set(rounds) == {"ROUND_1", "ROUND_2", "ROUND_3", "ROUND_4"}
    assert rounds["ROUND_4"] == data["ROUND_4"]


def test_validate_response_rejects_four_or_nearby_targets(tmp_path):
    masks_path = tmp_path / "masks.npz"
    masks = {}
    for key, position in {"1": (1, 1), "2": (3, 1), "3": (18, 18), "4": (1, 18)}.items():
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[position[1], position[0]] = 1
        masks[key] = mask
    np.savez(masks_path, **masks)
    loaded = np.load(masks_path)
    metadata = [{"candidate_id": int(key), "area": 1} for key in masks]
    rounds = {f"ROUND_{i}": {"short": str(i), "long": f"instruction {i}"} for i in range(1, 5)}
    with pytest.raises(ValueError, match="三个不同"):
        validate_response({"selected_ids": [1, 2, 3, 4], **rounds}, loaded, metadata)
    with pytest.raises(ValueError, match="距离过近"):
        validate_response({"selected_ids": [1, 2, 3], **rounds}, loaded, metadata)


def test_make_outputs_uses_global_fourth_mask_and_version(tmp_path):
    root = tmp_path / "IMAGE_MASK"
    (root / "IMAGE_2").mkdir(parents=True)
    Image.new("RGB", (8, 8), "gray").save(root / "IMAGE_2" / "sample.jpg")
    mask_arrays = {}
    for key, position in {"1": (0, 0), "2": (7, 0), "3": (0, 7)}.items():
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[position[1], position[0]] = 1
        mask_arrays[key] = mask
    masks_path = tmp_path / "masks.npz"
    np.savez(masks_path, **mask_arrays)
    masks = np.load(masks_path)
    rounds = {f"ROUND_{i}": {"short": str(i), "long": f"instruction {i}"} for i in range(1, 5)}
    metadata = {
        "candidates": [
            {"candidate_id": int(key), "area": 1} for key in mask_arrays
        ]
    }

    make_outputs(root, "sample", ["1", "2", "3"], rounds, metadata, masks)

    alpha = np.asarray(Image.open(root / "MASK" / "sample_MASK.png").getchannel("A")) < 128
    quadrants = [alpha[0:8, 0:8], alpha[0:8, 8:16], alpha[8:16, 0:8], alpha[8:16, 8:16]]
    assert [int(mask.sum()) for mask in quadrants] == [1, 2, 3, 64]
    selection_path = root / "SELECTION" / "sample_selection.json"
    assert is_current_selection(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["selection_mode"] == SELECTION_MODE


def test_old_selection_is_not_reused(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"selected_ids": [1, 2, 3, 4]}), encoding="utf-8")
    assert not is_current_selection(path)


def test_old_selection_is_regenerated_and_invalid_choice_retried(tmp_path):
    root = tmp_path / "IMAGE_MASK"
    for name in ("CANDIDATES", "IMAGE_2", "JSON", "MASK", "SELECTION"):
        (root / name).mkdir(parents=True)
    digest = "sample"
    Image.new("RGB", (20, 20), "gray").save(root / "IMAGE_2" / f"{digest}.jpg")
    Image.new("RGB", (20, 20), "white").save(
        root / "CANDIDATES" / f"{digest}_candidates.jpg"
    )
    arrays = {}
    for key, (x, y) in {"1": (1, 1), "2": (18, 1), "3": (1, 18), "4": (18, 18)}.items():
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[y, x] = 1
        arrays[key] = mask
    np.savez(root / "CANDIDATES" / f"{digest}_masks.npz", **arrays)
    metadata = {
        "candidates": [
            {"candidate_id": int(key), "area": int(key)} for key in arrays
        ]
    }
    meta_path = root / "CANDIDATES" / f"{digest}_meta.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    (root / "JSON" / f"{digest}_JSON_gpt.json").write_text("{}", encoding="utf-8")
    Image.new("RGBA", (40, 40), "white").save(root / "MASK" / f"{digest}_MASK.png")
    (root / "SELECTION" / f"{digest}_selection.json").write_text(
        json.dumps({"selected_ids": [1, 2, 3, 4]}), encoding="utf-8"
    )
    rounds = {f"ROUND_{index}": {"short": str(index), "long": f"instruction {index}"} for index in range(1, 5)}
    replies = [
        {"selected_ids": [1, 2, 3, 4], **rounds},
        {"selected_ids": [1, 2, 3], **rounds},
    ]

    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            payload = replies[self.calls]
            self.calls += 1
            message = SimpleNamespace(content=json.dumps(payload))
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    ok, _, _ = select_one(meta_path, root, client, "mock-model")
    assert ok
    assert completions.calls == 2
    assert is_current_selection(root / "SELECTION" / f"{digest}_selection.json")


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
    config = load_config(config_path)
    assert config.vision.api_key == "environment"
    assert config.vision.model == "local-model"
