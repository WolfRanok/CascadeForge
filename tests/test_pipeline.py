import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import cascadeforge.select as selection_module
from cascadeforge.config import load_config
from cascadeforge.organize import run_organize, split_grid
from cascadeforge.preprocess import crop_size, iou, select_candidates
from cascadeforge.select import (
    PROMPT,
    SELECTION_MODE,
    is_current_selection,
    make_outputs,
    pending_metadata,
    process_one as select_one,
    validate_response,
)


def test_crop_size_prefers_supported_ratio():
    assert crop_size(1600, 900) == (1600, 900, "16:9")


def test_selection_prompt_preserves_short_and_long_length_rules():
    assert "ROUND_1 至 ROUND_4" in PROMPT
    assert "short 均不超过 10 个汉字" in PROMPT
    assert "long 均为 15–30 个汉字" in PROMPT
    assert "明显、高对比、容易识别" in PROMPT
    assert "位置和当前外观明确的目标，清晰变化" in PROMPT
    assert "新状态必须不同" in PROMPT
    assert "蓝色物体改成蓝色" in PROMPT
    assert "左上部的白色蘑菇，尺寸明显变大一倍" in PROMPT


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
        [
            {"candidate_id": 1, "area": 1},
            {"candidate_id": 2, "area": 40},
            {"candidate_id": 3, "area": 20},
            {"candidate_id": 4, "area": 10},
        ],
    )
    assert selected == ["2", "3", "4", "1"]
    assert set(rounds) == {"ROUND_1", "ROUND_2", "ROUND_3", "ROUND_4"}
    assert rounds["ROUND_4"] == data["ROUND_1"]


def test_validate_response_rejects_three_or_nearby_targets(tmp_path):
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
    with pytest.raises(ValueError, match="四个不同"):
        validate_response({"selected_ids": [1, 2, 3], **rounds}, loaded, metadata)
    with pytest.raises(ValueError, match="距离过近"):
        validate_response({"selected_ids": [1, 2, 3, 4], **rounds}, loaded, metadata)


def test_make_outputs_uses_four_independent_target_masks(tmp_path):
    root = tmp_path / "IMAGE_MASK"
    (root / "IMAGE_2").mkdir(parents=True)
    Image.new("RGB", (8, 8), "gray").save(root / "IMAGE_2" / "sample.jpg")
    mask_arrays = {}
    for key, position in {"1": (0, 0), "2": (7, 0), "3": (0, 7), "4": (7, 7)}.items():
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

    make_outputs(root, "sample", ["1", "2", "3", "4"], rounds, metadata, masks)

    alpha = np.asarray(Image.open(root / "MASK" / "sample_MASK.png").getchannel("A")) < 128
    quadrants = [alpha[0:8, 0:8], alpha[0:8, 8:16], alpha[8:16, 0:8], alpha[8:16, 8:16]]
    assert [int(mask.sum()) for mask in quadrants] == [1, 1, 1, 1]
    selection_path = root / "SELECTION" / "sample_selection.json"
    assert is_current_selection(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["selection_mode"] == SELECTION_MODE


def test_old_selection_is_not_reused(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"selected_ids": [1, 2, 3, 4]}), encoding="utf-8")
    assert not is_current_selection(path)


def test_completed_legacy_selection_is_not_regenerated(tmp_path):
    root = tmp_path / "IMAGE_MASK"
    (root / "CANDIDATES").mkdir(parents=True)
    (root / "EDITED_4K").mkdir()
    (root / "CANDIDATES" / "sample_meta.json").write_text("{}", encoding="utf-8")
    Image.new("RGB", (8, 8), "gray").save(
        root / "EDITED_4K" / "sample_gpt_edited.jpg"
    )

    assert pending_metadata(root, set()) == []


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
        {"selected_ids": [1, 2, 3], **rounds},
        {"selected_ids": [1, 2, 3, 4], **rounds},
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


def test_run_selection_processes_new_rounds_and_does_not_repeat_failure(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "IMAGE_MASK"
    candidate_dir = root / "CANDIDATES"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "first_meta.json").write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        vision=SimpleNamespace(api_key="test-key", base_url="https://example.invalid", model="mock")
    )
    monkeypatch.setattr(selection_module, "load_config", lambda _: config)
    monkeypatch.setattr(selection_module, "OpenAI", lambda **_: object())
    sleeps = []
    monkeypatch.setattr(selection_module.time, "sleep", sleeps.append)
    calls = []

    def fake_process(meta_path, output_root, client, model):
        digest = meta_path.name.removesuffix("_meta.json")
        calls.append(digest)
        if digest == "first":
            # Simulate preprocessing producing another item during this round.
            (candidate_dir / "second_meta.json").write_text("{}", encoding="utf-8")
            return True, digest, "ok"
        return False, digest, "failed after internal retries"

    monkeypatch.setattr(selection_module, "process_one", fake_process)
    code = selection_module.run_selection(root, None, concurrency=1)

    assert code == 1
    assert calls == ["first", "second"]
    assert sleeps == [3]
    output = capsys.readouterr().out
    assert "第 2 轮" in output
    assert "成功 1，失败 1" in output


def test_run_selection_preserves_empty_directory_error(tmp_path, monkeypatch):
    root = tmp_path / "IMAGE_MASK"
    (root / "CANDIDATES").mkdir(parents=True)
    config = SimpleNamespace(
        vision=SimpleNamespace(api_key="test-key", base_url="https://example.invalid", model="mock")
    )
    monkeypatch.setattr(selection_module, "load_config", lambda _: config)
    assert selection_module.run_selection(root, None) == 1


def test_organize_splits_grid(tmp_path):
    edited = tmp_path / "abc_gpt_edited.jpg"
    source = tmp_path / "abc.jpg"
    image = Image.new("RGB", (8, 8), "white")
    image.save(edited)
    image.save(source)
    ok, digest, message = split_grid(edited, tmp_path / "out", source)
    assert ok and digest == "abc"
    assert (tmp_path / "out" / "abc" / "ROUND_4.jpg").exists()


def test_organize_copies_only_prompts_for_formal_results(tmp_path):
    root = tmp_path / "IMAGE_MASK"
    edited_dir = root / "EDITED_4K"
    json_dir = root / "JSON"
    source_dir = root / "IMAGE_2"
    for directory in (edited_dir, json_dir, source_dir):
        directory.mkdir(parents=True)
    for digest in ("ok1", "ok2", "failed"):
        Image.new("RGB", (8, 8), "white").save(edited_dir / f"{digest}_gpt_edited.jpg")
        Image.new("RGB", (4, 4), "white").save(source_dir / f"{digest}.jpg")
        (json_dir / f"{digest}_JSON_gpt.json").write_text(
            json.dumps({"ROUND_1": {"long": digest}}), encoding="utf-8"
        )
    output_dir = tmp_path / "OUTPUT" / "AC_multi_object"
    prompt_dir = tmp_path / "OUTPUT" / "提示词"
    (prompt_dir / "stale.json").parent.mkdir(parents=True)
    (prompt_dir / "stale.json").write_text("{}", encoding="utf-8")
    # Only these two images represent successful edit results for this run.
    (edited_dir / "failed_gpt_edited.jpg").unlink()

    assert run_organize(edited_dir, output_dir, source_dir, workers=1) == 0
    assert sorted(path.name for path in prompt_dir.glob("*.json")) == ["ok1.json", "ok2.json"]
    assert (output_dir / "ok1" / "ROUND_4.jpg").exists()
    assert (output_dir / "ok2" / "ROUND_4.jpg").exists()
    assert not (prompt_dir / "ok1_JSON_gpt.json").exists()


def test_organize_skips_result_without_prompt_and_returns_failure(tmp_path):
    root = tmp_path / "IMAGE_MASK"
    edited_dir = root / "EDITED_4K"
    source_dir = root / "IMAGE_2"
    edited_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(edited_dir / "missing_gpt_edited.jpg")
    Image.new("RGB", (4, 4), "white").save(source_dir / "missing.jpg")

    output_dir = tmp_path / "OUTPUT" / "AC_multi_object"
    assert run_organize(edited_dir, output_dir, source_dir, workers=1) == 1
    assert not (output_dir / "missing").exists()


def test_config_environment_overrides_local(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"moliapi": {"api_key": "local", "model": "local-model"}}), encoding="utf-8")
    monkeypatch.setenv("GPT_API_KEY", "environment")
    monkeypatch.setenv("GPT_MODEL", "local-model")
    config = load_config(config_path)
    assert config.vision.api_key == "environment"
    assert config.vision.model == "local-model"
