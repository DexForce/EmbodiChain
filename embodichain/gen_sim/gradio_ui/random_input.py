# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

import base64
import json
import os
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from embodichain.gen_sim.env import load_gen_sim_env

load_gen_sim_env()

EMBODICHAIN_ROOT = Path(
    os.environ.get("EMBODICHAIN_ROOT") or str(Path(__file__).resolve().parents[3])
).expanduser()
APP_ROOT = Path(__file__).resolve().parent
IMAGE_DIR = Path(
    os.environ.get(
        "AUTO_IMAGE_DIR",
        str(EMBODICHAIN_ROOT / "gym_project/action_agent_pipeline/auto_images"),
    )
).expanduser()
AUTO_IMAGE_DIR_IS_CONFIGURED = "AUTO_IMAGE_DIR" in os.environ
FALLBACK_IMAGE_DIR = (
    EMBODICHAIN_ROOT / "gym_project/action_agent_pipeline/baseline_image_input"
)
PREBUILT_SCENE_DIR = Path(
    os.environ.get("AUTO_PREBUILT_SCENE_DIR", str(APP_ROOT / "scenes"))
).expanduser()
GENERATED_IMAGE_DIR = Path(
    os.environ.get("AUTO_GENERATED_IMAGE_DIR", "./tmp_img/auto")
).expanduser()
IMAGE_API_KEY = os.environ.get("AUTO_IMAGE_API_KEY", os.environ.get("ARK_API_KEY", ""))
IMAGE_API_URL = os.environ.get(
    "AUTO_IMAGE_API_URL",
    "https://ark.cn-beijing.volces.com/api/v3",
)
IMAGE_MODEL = os.environ.get("AUTO_IMAGE_MODEL", "doubao-seedream-4-5-251128")
IMAGE_SIZE = os.environ.get("AUTO_IMAGE_SIZE", "2848x1600")
IMAGE_PROMPT = (
    "以原图为基础参考，严格保持原图的相机视角、拍摄距离、透视关系、画面构图、背景环境、桌面材质、桌面纹理、光影方向、阴影位置和整体明暗关系。"
    "原图中已有物体的类别、大小、形状、轮廓、空间位置、朝向、部件数量和结构比例必须保持不变。"
    "如果提供了 Scene description，只根据该描述在桌面上新增对应背景物体；新增物体必须放在描述指定的位置，尺寸、透视、遮挡和阴影要与原图自然一致。"
    "不要删除原有物体，不要移动原有物体，不要改变主任务物体的结构。高清细节，真实自然。"
)

TASK_DESCRIPTIONS: dict[tuple[int, int], str] = {
    (0, 0): "用双臂把两侧的罐头和瓶子放到篮子里",
    (0, 1): "用双臂把两侧的方块放到篮子里",
    (0, 2): "用双臂把两侧的方块和纸杯放到篮子里",
    (0, 3): "用双臂把两侧的方块和苹果放到篮子里",
    (1, 0): "用双臂把塑料水盆往前移动",
    (1, 1): "用双臂把木棍往前移动",
    (1, 2): "用双臂往把苹果和魔方放入盘子，然后用双臂端起盘子",
    (1, 3): "用双臂把托盘往前移动",
    (2, 0): "用双臂把两侧的香蕉放到盘子里，然后用双臂端起盘子",
    (2, 1): "用双臂把两侧的罐头扶正",
    (2, 2): "用双臂把两侧的瓶子和罐头扶正",
    (2, 3): "用双臂把两侧的罐头扶正",
    (3, 0): "把桌面上的物体按照方块按照从右往左的顺序叠起来",
    (3, 1): "把桌面上的物体按照右边的方块，左边的方块，纸杯的顺序叠起来",
    (3, 2): "把纸杯叠放到爆米花桶上，把蓝色耳机叠放到爆米花桶上",
    (3, 3): "把纸杯叠放到爆米花桶上，把固体胶叠放到爆米花桶上",
    (4, 0): "把桌面上的方块摆成一排",
    (4, 1): "把桌面上的物体按照瓶子，方块排成一排",
    (4, 2): "把桌面上的罐头摆成一排",
    (4, 3): "把桌面上的物体按照瓶子，罐头，方块的顺序摆成一排",
}

TASK_DESCRIPTIONS_EN: dict[tuple[int, int], str] = {
    (
        0,
        0,
    ): "Use both arms to place the cans and bottles on both sides into the basket.",
    (0, 1): "Use both arms to place the blocks on both sides into the basket.",
    (
        0,
        2,
    ): "Use both arms to place the blocks and paper cups on both sides into the basket.",
    (
        0,
        3,
    ): "Use both arms to place the blocks and apples on both sides into the basket.",
    (1, 0): "Use both arms to move the plastic basin forward.",
    (1, 1): "Use both arms to move the wooden stick forward.",
    (
        1,
        2,
    ): "Use both arms to place the apple and Rubik's Cube onto the plate, then use both arms to lift the plate.",
    (1, 3): "Use both arms to move the tray forward.",
    (
        2,
        0,
    ): "Use both arms to place the bananas on both sides onto the tray, then use both arms to lift the tray.",
    (2, 1): "Use both arms to set the cans on both sides upright.",
    (2, 2): "Use both arms to set the bottles and cans on both sides upright.",
    (2, 3): "Use both arms to set the cans on both sides upright.",
    (3, 0): "Stack the blocks on the table in order from right to left.",
    (
        3,
        1,
    ): "Stack the objects on the table in this order: right block, left block, paper cup.",
    (
        3,
        2,
    ): "Stack the paper cup on the popcorn bucket, then stack the blue headphones on the popcorn bucket.",
    (
        3,
        3,
    ): "Stack the paper cup on the popcorn bucket, then stack the glue stick on the popcorn bucket.",
    (4, 0): "Arrange the blocks on the table in a row.",
    (4, 1): "Arrange the objects on the table in a row in this order: bottle, block.",
    (4, 2): "Arrange the cans on the table in a row.",
    (
        4,
        3,
    ): "Arrange the objects on the table in a row in this order: bottle, can, block.",
}

RELATION_PATTERN = {
    (0, 0): ["at the left side of the can", "at the right side of the bottle"],
    (0, 1): [
        "at the left side of the left cheese cube",
        "at the right side of the right cheese cube",
    ],
    (0, 2): ["at the left side of the cube", "at the right side of the cup"],
    (0, 3): ["at the left side of the cube", "at the right side of the apple"],
    (1, 0): [],
    (1, 1): [],
    (1, 2): [],
    (1, 3): [],
    (2, 0): [
        "at the left side of the left bottle",
        "at the right side of the right bottle",
    ],
    (2, 1): [
        "at the left side of the left soda can",
        "at the right side of the right soda can",
    ],
    (2, 2): ["at the left side of the bottle", "at the right side of the can"],
    (2, 3): ["at the left side of the paper cup", "at the right side of the soda can"],
    (3, 0): [],
    (3, 1): [],
    (3, 2): [],
    (3, 3): [],
    (4, 0): [],
    (4, 1): [],
    (4, 2): [],
    (4, 3): [],
}

AREA_PATTERN = [
    "at the left side of the table",
    "at the right side of the table",
    "at the front of the table",
    "at the front right corner of the table",
    "at the front left corner of the table",
]

OBJECT_LIST = [
    "cup",
    "potted plant",
    "clock",
    "book",
    "pen",
    "bottle",
    "soda can",
    "photo frame",
    "apple",
    "peach",
    "bread",
    "chocolate bar",
    "cookie",
    "penholder",
    "desk lamp",
    "stapler",
    "headphones",
    "desk calendar",
    "eyeglasses",
    "fan",
    "bluetooth speaker",
    "table mirror",
    "computer mouse",
    "keyboard",
]

CHINESE_OBJECT_NAMES = {
    "cup": "杯子",
    "potted plant": "盆栽",
    "clock": "时钟",
    "book": "书",
    "bottle": "瓶子",
    "soda can": "易拉罐",
    "photo frame": "相框",
    "apple": "苹果",
    "peach": "桃子",
    "bread": "小面包",
    "chocolate bar": "巧克力棒",
    "cookie": "饼干",
    "penholder": "笔筒",
    "desk lamp": "小台灯",
    "stapler": "订书机",
    "headphones": "耳机",
    "small desk calendar": "小台历",
    "eyeglasses": "眼镜",
    "fan": "小风扇",
    "bluetooth speaker": "蓝牙音箱",
    "computer mouse": "鼠标",
}


CHINESE_SPATIAL_RELATIONS = {
    "at the left side of the can": "罐头左侧",
    "at the right side of the bottle": "瓶子右侧",
    "at the left side of the left cheese cube": "左侧奶酪方块左侧",
    "at the right side of the right cheese cube": "右侧奶酪方块右侧",
    "at the left side of the cube": "方块左侧",
    "at the right side of the cup": "杯子右侧",
    "at the right side of the apple": "苹果右侧",
    "at the left side of the left bottle": "左侧瓶子左侧",
    "at the right side of the right bottle": "右侧瓶子右侧",
    "at the left side of the left soda can": "左侧易拉罐左侧",
    "at the right side of the right soda can": "右侧易拉罐右侧",
    "at the left side of the bottle": "瓶子左侧",
    "at the right side of the can": "罐头右侧",
    "at the left side of the paper cup": "纸杯左侧",
    "at the right side of the soda can": "易拉罐右侧",
    "at the left side of the table": "桌子左侧",
    "at the right side of the table": "桌子右侧",
    "at the front of the table": "桌子前侧",
    "at the front right corner of the table": "桌子右前角",
    "at the front left corner of the table": "桌子左前角",
    "on the table": "桌面上",
}


@dataclass(frozen=True)
class AutoInput:
    task_index: tuple[int, int]
    base_image_path: Path
    prebuilt_scene_dir: Path | None
    image_path: Path | None
    task_description: str
    scene_description: str

    def to_json_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["task_index"] = list(self.task_index)
        value["base_image_path"] = self.base_image_path.as_posix()
        value["prebuilt_scene_dir"] = (
            self.prebuilt_scene_dir.as_posix() if self.prebuilt_scene_dir else None
        )
        value["image_path"] = self.image_path.as_posix() if self.image_path else None
        return value


def task_id(task_index: tuple[int, int]) -> str:
    return f"task{task_index[0]}_{task_index[1]}"


def parse_task_id(value: str) -> tuple[int, int] | None:
    stem = Path(value).stem
    if not stem.startswith("task"):
        return None
    parts = stem[4:].split("_", maxsplit=1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def auto_image_directories() -> tuple[Path, ...]:
    """Return image sources in precedence order for the Auto loop.

    A user-supplied ``AUTO_IMAGE_DIR`` is authoritative.  With the default
    directory, retain compatibility with deployments that have the checked-in
    ``baseline_image_input`` set but have not created ``auto_images`` yet.
    """
    directories = [IMAGE_DIR]
    if not AUTO_IMAGE_DIR_IS_CONFIGURED and FALLBACK_IMAGE_DIR != IMAGE_DIR:
        directories.append(FALLBACK_IMAGE_DIR)
    return tuple(directories)


def available_auto_task_indices() -> tuple[tuple[int, int], ...]:
    """Return only task variants whose input image and clean scene can be resolved."""
    return tuple(
        task_index
        for task_index in TASK_DESCRIPTIONS
        if any(
            (image_dir / f"{task_id(task_index)}.png").is_file()
            for image_dir in auto_image_directories()
        )
        and get_prebuilt_scene_dir(task_index).is_dir()
    )


def random_task(rng: np.random.Generator) -> tuple[int, int]:
    available_tasks = available_auto_task_indices()
    if not available_tasks:
        expected = ", ".join(str(path) for path in auto_image_directories())
        raise FileNotFoundError(
            "No Auto input images were found. Add task<task>_<sub_task>.png "
            f"files to: {expected}"
        )
    return available_tasks[int(rng.integers(0, len(available_tasks)))]


def get_base_image_path(task_index: tuple[int, int]) -> Path:
    filename = f"{task_id(task_index)}.png"
    for image_dir in auto_image_directories():
        candidate = image_dir / filename
        if candidate.is_file():
            return candidate
    return IMAGE_DIR / filename


def get_prebuilt_scene_dir(task_index: tuple[int, int]) -> Path:
    return PREBUILT_SCENE_DIR / task_id(task_index)


def get_task_description(task_index: tuple[int, int], *, language: str = "zh") -> str:
    descriptions = TASK_DESCRIPTIONS_EN if language == "en" else TASK_DESCRIPTIONS
    try:
        return descriptions[task_index]
    except KeyError as exc:
        raise KeyError(f"No task description configured for task{task_index}") from exc


def image_to_base64(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    else:
        raise ValueError(f"Not supported: {ext}, only jpg/jpeg/png are supported")
    with path.open("rb") as file:
        b64_str = base64.b64encode(file.read()).decode("utf-8")
    return f"data:{mime};base64,{b64_str}"


def build_image_prompt(scene_description: str = "") -> str:
    scene_description = (scene_description or "").strip()
    if not scene_description:
        return IMAGE_PROMPT
    return (
        f"{IMAGE_PROMPT}\n\n"
        "Scene description:\n"
        f"{scene_description}\n\n"
        "严格执行 Scene description 中的新增物体和空间位置要求。"
    )


def create_image_input(
    task_index: tuple[int, int],
    *,
    scene_description: str = "",
    output_dir: Path = GENERATED_IMAGE_DIR,
) -> Path:
    base_image_path = get_base_image_path(task_index)
    if not base_image_path.is_file():
        raise FileNotFoundError(f"Base auto image not found: {base_image_path}")

    from volcenginesdkarkruntime import Ark
    import requests

    image_base64 = image_to_base64(base_image_path)
    client = Ark(api_key=IMAGE_API_KEY, base_url=IMAGE_API_URL)
    response = client.images.generate(
        model=IMAGE_MODEL,
        prompt=build_image_prompt(scene_description),
        image=image_base64,
        size=IMAGE_SIZE,
        response_format="url",
        watermark=False,
    )
    image_url = response.data[0].url
    resp = requests.get(image_url, timeout=60)
    resp.raise_for_status()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"auto_task{task_index[0]}_{task_index[1]}_{uuid.uuid4().hex[:12]}.png"
    )
    output_path.write_bytes(resp.content)
    return output_path


def create_text_input(
    task_index: tuple[int, int],
    rng: np.random.Generator,
    *,
    language: str = "en",
    min_background_objects: int = 0,
) -> str:
    text_parts: list[str] = []
    if task_index[0] == 5 and min_background_objects == 0:
        return ""

    mu, sigma = 1.0, 1.0
    raw = rng.normal(loc=mu, scale=sigma)
    num_background_objects = int(np.clip(np.round(raw), 0, 3))
    if min_background_objects > 0:
        num_background_objects = max(num_background_objects, min_background_objects)

    if num_background_objects == 0:
        return ""

    selected_objects = rng.choice(
        OBJECT_LIST,
        size=num_background_objects,
        replace=False,
    ).tolist()

    spatial_candidates = []
    spatial_candidates.extend(RELATION_PATTERN.get(task_index, []))
    spatial_candidates.extend(AREA_PATTERN)
    spatial_candidates.append("on the table")

    for obj in selected_objects:
        selected_spatial = rng.choice(spatial_candidates)
        if language == "zh":
            chinese_object = CHINESE_OBJECT_NAMES.get(obj, obj)
            chinese_relation = CHINESE_SPATIAL_RELATIONS.get(
                selected_spatial,
                selected_spatial,
            )
            text_parts.append(f"将一个{chinese_object}放在{chinese_relation}。")
        else:
            article = "an" if obj[0].lower() in {"a", "e", "i", "o", "u"} else "a"
            text_parts.append(f"Place {article} {obj} {selected_spatial}.")

    return " ".join(text_parts)


def generate_auto_scene_description(
    *,
    rng: np.random.Generator | None = None,
    task_index: tuple[int, int] | None = None,
    language: str = "en",
    ensure_scene: bool = False,
) -> str:
    rng = rng or np.random.default_rng()
    task_index = task_index or random_task(rng)
    return create_text_input(
        task_index,
        rng,
        language=language,
        min_background_objects=1 if ensure_scene else 0,
    )


def generate_auto_text_input(
    *,
    rng: np.random.Generator | None = None,
    task_index: tuple[int, int] | None = None,
    language: str = "en",
    ensure_scene: bool = False,
    include_scene: bool = True,
) -> AutoInput:
    rng = rng or np.random.default_rng()
    task_index = task_index or random_task(rng)
    base_image_path = get_base_image_path(task_index)
    prebuilt_scene_dir = get_prebuilt_scene_dir(task_index)
    if not base_image_path.is_file():
        raise FileNotFoundError(f"Base auto image not found: {base_image_path}")
    if not prebuilt_scene_dir.is_dir():
        raise FileNotFoundError(f"Prebuilt scene not found: {prebuilt_scene_dir}")
    return AutoInput(
        task_index=task_index,
        base_image_path=base_image_path,
        prebuilt_scene_dir=prebuilt_scene_dir,
        image_path=None,
        task_description=get_task_description(task_index, language=language),
        scene_description=(
            generate_auto_scene_description(
                rng=rng,
                task_index=task_index,
                language=language,
                ensure_scene=ensure_scene,
            )
            if include_scene
            else ""
        ),
    )


def generate_auto_image(
    auto_input: AutoInput,
    *,
    output_dir: Path = GENERATED_IMAGE_DIR,
) -> AutoInput:
    image_path = create_image_input(
        auto_input.task_index,
        scene_description=auto_input.scene_description,
        output_dir=output_dir,
    )
    return replace(auto_input, image_path=image_path)


def generate_auto_input(
    *,
    rng: np.random.Generator | None = None,
    task_index: tuple[int, int] | None = None,
    output_dir: Path = GENERATED_IMAGE_DIR,
    language: str = "en",
) -> AutoInput:
    auto_input = generate_auto_text_input(
        rng=rng,
        task_index=task_index,
        language=language,
    )
    return generate_auto_image(auto_input, output_dir=output_dir)


def main() -> None:
    auto_input = generate_auto_input()
    print(json.dumps(auto_input.to_json_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
