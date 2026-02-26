# /root/dyadic-interact/src/data_preprocess/generate_splits.py

from __future__ import annotations

from typing import Any, Dict, List, Sequence
import json
from collections import defaultdict
from pathlib import Path
import os

import numpy as np
import tqdm
from tqdm.contrib.concurrent import process_map
from icecream import ic

from configs import SMPLX_NPZ_FOLDER, FPS, N_FRAMES_DELTA_THRESHOLD, DATASET_ROOT_DIR


SPLITS: Sequence[str] = ("train", "dev", "test")
INTERACTION_TYPES: Sequence[str] = ("naturalistic", "improvised")

SPLITS_RATIO_PATH = "configs/splits_ratio.json"


def _parse_ratio_field(mapping: Dict[str, Any], key: str) -> Dict[str, float]:
    """
    把形如 "train:dev:test": "8:1:1" 解析成 {"train": 0.8, "dev": 0.1, "test": 0.1}
    """
    raw = mapping[key]
    parts = [int(x) for x in raw.split(":")]
    ratios = [x / sum(parts) for x in parts]

    keys = key.split(":")
    return dict(zip(keys, ratios))


def get_label_duration_dict(
    splits_ratio_path: str = SPLITS_RATIO_PATH,
) -> Dict[str, Dict[str, float]]:
    """
    读取 `configs/splits_ratio.json`，返回各 split / interaction_type 对应的小时数。

    返回结构示例：
    {
        "train": {"naturalistic": 60.0, "improvised": 30.0},
        "dev":   {"naturalistic": 20.0, "improvised": 10.0},
        "test":  {"naturalistic": 20.0, "improvised": 10.0},
    }
    """
    with open(splits_ratio_path, "r") as f:
        json_dict = json.load(f)

    total_duration: float = json_dict["total_duration"]

    split_ratio_dict = _parse_ratio_field(json_dict, "train:dev:test")
    interaction_ratio_dict = _parse_ratio_field(
        json_dict, "naturalistic:improvised")

    label_duration_dict: Dict[str, Dict[str, float]] = defaultdict(dict)
    for interaction_type in INTERACTION_TYPES:
        for split in SPLITS:
            n_hours = (
                total_duration
                * interaction_ratio_dict[interaction_type]
                * split_ratio_dict[split]
            )
            label_duration_dict[split][interaction_type] = n_hours

    return label_duration_dict


def get_file_info(npz_path: str) -> Dict[str, Any] | None:
    """
    从单个 npz 文件提取基本信息：
    - 帧数（各数组最小帧数）
    - interaction_type / split
    - file_id / file_short_path / file_id_prefix

    如果同一个 npz 中各数组帧数差异超过阈值，则返回 None（表示丢弃该文件）。
    """
    data = dict(np.load(npz_path, allow_pickle=True))

    n_frame_list = [v.shape[0] for v in data.values()]
    if max(n_frame_list) - min(n_frame_list) > N_FRAMES_DELTA_THRESHOLD:
        return None

    n_frame = min(n_frame_list)

    file_name = os.path.basename(npz_path).replace("_smplx.npz", "")
    label = os.path.dirname(npz_path).split(
        "/")[-1]  # 例如: "naturalistic_train_batch0000"
    interaction_type, split = label.split("_")[:2]

    return {
        "file_id": file_name,
        "file_short_path": label + "/" + file_name,
        "interaction_type": interaction_type,
        "split": split,
        "n_frame": n_frame,
        "file_id_prefix": "_".join(file_name.split("_")[:-1]),
    }


def build_file_list(npz_path_list: Sequence[str]) -> List[Dict[str, Any]]:
    """
    使用 process_map 并行解析所有 npz，过滤掉返回 None 的无效项。
    """
    file_list = process_map(
        get_file_info,
        npz_path_list,
        desc="get file info",
        chunksize=32,  # 避免 TqdmWarning: Iterable length > 1000 but chunksize is not set
    )
    return [f for f in file_list if f is not None]


def build_paired_file_list(
    file_list: Sequence[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """
    按 `file_id_prefix` 分组，并只保留“恰好有两条记录”的 prefix 作为有效 pair。

    返回：每个元素是长度为 2 的 list，形如：
    [
        {"file_id": "..._P0739", ...},
        {"file_id": "..._P0740", ...},
    ]
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for info in file_list:
        prefix = info["file_id_prefix"]
        grouped[prefix].append(info)

    paired_file_list: List[List[Dict[str, Any]]] = []
    for prefix, files in grouped.items():
        if len(files) == 2:
            if abs(files[0]['n_frame'] - files[1]['n_frame']) > N_FRAMES_DELTA_THRESHOLD:
                continue
            paired_file_list.append(files)

    return paired_file_list


def select_pairs_for_duration(
    paired_file_list: Sequence[List[Dict[str, Any]]],
    split: str,
    interaction_type: str,
    duration_hours: float,
) -> List[List[Dict[str, Any]]]:
    """
    在给定的 split / interaction_type 下，按顺序选择若干 pair，
    直到总帧数 >= duration_hours * FPS * 3600。

    注意：这里假设 pair 中两条记录帧数一致，因此用 n_frame * 2。
    """
    filtered_pairs = [
        pair
        for pair in paired_file_list
        if pair[0]["split"] == split
        and pair[0]["interaction_type"] == interaction_type
    ]

    needed_frames = duration_hours * FPS * 3600

    total_n_frame = 0
    last_index = -1

    for i, pair in enumerate(filtered_pairs):
        # 每个 pair 有两条等长记录，这里用第一条的 n_frame * 2 近似总帧数
        total_n_frame += pair[0]["n_frame"] * 2
        if total_n_frame > needed_frames:
            last_index = i
            break

    if total_n_frame < needed_frames or last_index == -1:
        raise ValueError(
            f"total_n_frame: {total_n_frame} is less than needed_frames: {needed_frames}"
        )

    return list(filtered_pairs[: last_index + 1])


def build_split_dict(
    paired_file_list: Sequence[List[Dict[str, Any]]],
    label_duration_dict: Dict[str, Dict[str, float]],
) -> Dict[str, List[List[Dict[str, Any]]]]:
    """
    按照 label_duration_dict 规定的小时数，为每个 split / interaction_type
    选取对应数量的 pairs，组装成最终的 split_dict。
    """
    split_dict: Dict[str, List[List[Dict[str, Any]]]] = {}

    for split in SPLITS:
        split_pairs: List[List[Dict[str, Any]]] = []
        for interaction_type in INTERACTION_TYPES:
            duration = label_duration_dict[split][interaction_type]
            selected_pairs = select_pairs_for_duration(
                paired_file_list, split, interaction_type, duration
            )
            split_pairs.extend(selected_pairs)
        split_dict[split] = split_pairs

    return split_dict


def save_split_jsons(
    split_dict: Dict[str, List[List[Dict[str, Any]]]],
    out_dir: str | Path,
) -> None:
    """
    将 split_dict 保存成 3 个 json 文件，每个文件是一个 list：
    [
        [file_short_path_1, file_short_path_2],
        ...
    ]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        pairs = split_dict[split]
        pair_paths = [
            [pair[0]["file_short_path"], pair[1]["file_short_path"]]
            for pair in pairs
        ]
        out_path = out_dir / f"{split}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(pair_paths, f, indent=4, ensure_ascii=False)


def summarize_and_log(
    npz_path_list: Sequence[str],
    file_list: Sequence[Dict[str, Any]],
    paired_file_list: Sequence[List[Dict[str, Any]]],
    split_dict: Dict[str, List[List[Dict[str, Any]]]],
    label_duration_dict: Dict[str, Dict[str, float]],
) -> None:
    """
    使用 icecream 打印一些统计信息，方便 sanity check / review。
    """
    for split in SPLITS:
        ic(split)
        ic(len(split_dict[split]))

    ic(label_duration_dict)
    ic(len(npz_path_list))
    ic(len(file_list))

    file_ids = [f["file_id"] for f in file_list]
    ic(len(file_ids))

    file_id_prefix_list = list({f["file_id_prefix"] for f in file_list})
    ic(len(file_id_prefix_list))

    ic(len(paired_file_list))
    if paired_file_list:
        ic(paired_file_list[0])

    total_frames = sum(pair[0]["n_frame"] * 2 for pair in paired_file_list)
    ic(total_frames / FPS / 3600)


def main() -> None:
    # 1. 收集所有 npz 路径
    npz_path_list = sorted(
        str(p) for p in Path(SMPLX_NPZ_FOLDER).rglob("*_smplx.npz")
    )

    # 2. 并行解析文件信息
    file_list = build_file_list(npz_path_list)

    # 3. 校验 file_id 唯一性
    file_ids = [f["file_id"] for f in file_list]
    if len(file_ids) != len(set(file_ids)):
        raise ValueError(
            f"file_ids is not unique, len={len(file_ids)}, unique={len(set(file_ids))}")

    # 4. 根据 file_id_prefix 构建成对的样本
    paired_file_list = build_paired_file_list(file_list)

    # 5. 计算每个 split / interaction_type 需要的时长（小时）
    label_duration_dict = get_label_duration_dict()

    # 6. 根据时长需求选择对应的 pairs
    split_dict = build_split_dict(paired_file_list, label_duration_dict)

    # 7. 打印统计信息，方便检查
    summarize_and_log(
        npz_path_list=npz_path_list,
        file_list=file_list,
        paired_file_list=paired_file_list,
        split_dict=split_dict,
        label_duration_dict=label_duration_dict,
    )

    # 8. 保存 json（例如保存在 DATASET_ROOT_DIR/splits/ 目录下）

    save_split_jsons(split_dict, out_dir=DATASET_ROOT_DIR)


if __name__ == "__main__":
    main()
