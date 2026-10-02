"""Work handoff: prepare bounded source chunks, validate drafts, render a course."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

from .doc import _time_link
from .subs import to_srt

SCHEMA = 1
MAX_SECONDS = 600
MAX_CHARS = 9000


class DraftError(ValueError):
    """An incomplete or stale Work draft must never become a published course."""


def _json(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DraftError(f"无法读取 {path}：{exc}") from exc


def _text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DraftError(f"{field} 必须是非空字符串")
    return value.strip()


def _segments(raw: list[dict]) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise DraftError("字幕为空，无法准备素材")
    out = []
    for i, seg in enumerate(raw, 1):
        if not isinstance(seg, dict):
            raise DraftError(f"字幕第 {i} 条不是对象")
        try:
            start, end = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DraftError(f"字幕第 {i} 条没有有效时间") from exc
        if not (math.isfinite(start) and math.isfinite(end) and
                0 <= start <= end and (not out or start >= out[-1]["start"])):
            raise DraftError(f"字幕第 {i} 条时间无效或顺序错误")
        out.append({"id": f"s{i:06d}", "start": start, "end": end,
                    "text": _text(seg.get("text"), f"字幕第 {i} 条文本")})
    return out


def _chunk_segments(segs: list[dict], max_seconds: int, max_chars: int):
    if max_seconds <= 0 or max_chars <= 0:
        raise DraftError("分块时长和字符数必须大于 0")
    chunks, current, length = [], [], 0
    for seg in segs:
        if current and (seg["end"] - current[0]["start"] > max_seconds or
                        length + len(seg["text"]) > max_chars):
            chunks.append(current)
            current, length = [], 0
        current.append(seg)
        length += len(seg["text"])
    if current:
        chunks.append(current)
    return chunks


def prepare_bundle(work: Path, title: str, src: str, raw: list[dict],
                   frames: list[dict] | None = None, input_name: str = "subs.json",
                   max_seconds: int = MAX_SECONDS, max_chars: int = MAX_CHARS) -> dict:
    """Write model-free inputs and editable draft templates; preserve existing drafts."""
    work = Path(work)
    title = _text(title, "课程标题")
    segs = _segments(raw)
    cuts = _chunk_segments(segs, max_seconds, max_chars)
    frame_rows = []
    for i, frame in enumerate(frames or [], 1):
        path = Path(frame["path"])
        if not path.is_file() or not path.resolve().is_relative_to(work.resolve()):
            raise DraftError(f"第 {i} 张候选帧不在课程目录内：{path}")
        t = float(frame["t"])
        if not math.isfinite(t) or t < 0:
            raise DraftError(f"第 {i} 张候选帧时间无效")
        frame_rows.append({"id": f"f{i:04d}", "t": t,
                           "path": path.resolve().relative_to(work.resolve()).as_posix()})

    chunk_dir, draft_dir = work / "chunks", work / "draft"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    draft_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, group in enumerate(cuts, 1):
        chunk_id = f"chunk-{i:03d}"
        first, last = group[0]["start"], group[-1]["end"]
        nearby = [f for f in frame_rows if first <= f["t"] <= last]
        payload = {"schema": SCHEMA, "chunk_id": chunk_id, "title": title,
                   "source": src, "segments": group, "frames": nearby}
        path = chunk_dir / f"{chunk_id}.json"
        path.write_text(_json(payload), encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        draft_path = draft_dir / f"{chunk_id}.json"
        if not draft_path.exists():
            draft_path.write_text(_json({"chunk_id": chunk_id,
                                         "source_sha256": digest,
                                         "paragraphs": []}), encoding="utf-8")
        entries.append({"id": chunk_id, "input": path.relative_to(work).as_posix(),
                        "draft": draft_path.relative_to(work).as_posix(),
                        "source_sha256": digest, "first_id": group[0]["id"],
                        "last_id": group[-1]["id"], "start": first, "end": last,
                        "segments": len(group), "frames": len(nearby)})

    meta = draft_dir / "metadata.json"
    if not meta.exists():
        meta.write_text(_json({"title": title, "intro": ""}), encoding="utf-8")
    manifest = {"schema": SCHEMA, "title": title, "source": src,
                "input": input_name, "segments": len(segs), "frames": frame_rows,
                "chunks": entries}
    (work / "manifest.json").write_text(_json(manifest), encoding="utf-8")
    (work / "transcript.input.srt").write_text(
        to_srt(raw), encoding="utf-8")
    return manifest


def _inside(work: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise DraftError("清单中的文件路径无效")
    path = (work / relative).resolve()
    if not path.is_relative_to(work.resolve()):
        raise DraftError(f"文件路径越界：{relative}")
    return path


def validate_bundle(work: Path) -> tuple[dict, dict, list[dict]]:
    """Check exact source coverage, chunk identity and selected frame placement."""
    work = Path(work)
    manifest = _read(work / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise DraftError("manifest.json 版本不匹配，请重新 prepare")
    meta = _read(work / "draft" / "metadata.json")
    if not isinstance(meta, dict):
        raise DraftError("draft/metadata.json 必须是对象")
    _text(meta.get("title"), "标题")
    _text(meta.get("intro"), "课程导语")
    entries = manifest.get("chunks")
    if not isinstance(entries, list) or not entries:
        raise DraftError("清单中没有分块")
    frames = manifest.get("frames", [])
    if not isinstance(frames, list):
        raise DraftError("候选帧清单无效")
    frame_by_id = {f["id"]: f for f in frames}
    if len(frame_by_id) != len(frames):
        raise DraftError("候选帧 ID 重复")
    paragraphs, seen_frames = [], set()
    total = 0
    previous_end = -1.0
    for entry in entries:
        chunk_id = entry["id"]
        source_path = _inside(work, entry["input"])
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if digest != entry["source_sha256"]:
            raise DraftError(f"{chunk_id} 素材已变更，请重新 prepare")
        chunk, draft = _read(source_path), _read(_inside(work, entry["draft"]))
        if (not isinstance(chunk, dict) or not isinstance(draft, dict) or
                chunk.get("chunk_id") != chunk_id or draft.get("chunk_id") != chunk_id or
                draft.get("source_sha256") != digest):
            raise DraftError(f"{chunk_id} 草稿与当前素材不匹配")
        segs = chunk.get("segments")
        if not isinstance(segs, list) or not segs:
            raise DraftError(f"{chunk_id} 没有有效字幕")
        ids = [s["id"] for s in segs]
        expected_ids = [f"s{i:06d}" for i in range(total + 1, total + len(segs) + 1)]
        if (ids != expected_ids or ids[0] != entry["first_id"] or
                ids[-1] != entry["last_id"] or segs[0]["start"] < previous_end):
            raise DraftError(f"{chunk_id} 与上一块之间的字幕 ID 或时间不连续")
        items = draft.get("paragraphs")
        if not isinstance(items, list) or not items:
            raise DraftError(f"{chunk_id} 尚未写稿")
        cursor = 0
        for j, item in enumerate(items, 1):
            if not isinstance(item, dict):
                raise DraftError(f"{chunk_id} 第 {j} 段不是对象")
            start, end = item.get("start_id"), item.get("end_id")
            if start not in ids or end not in ids:
                raise DraftError(f"{chunk_id} 第 {j} 段引用了不存在的字幕 ID")
            a, b = ids.index(start), ids.index(end)
            if a != cursor or b < a:
                expected = ids[cursor] if cursor < len(ids) else "没有更多字幕"
                raise DraftError(f"{chunk_id} 第 {j} 段字幕范围有缺失或重复；期望从 {expected} 开始")
            text = _text(item.get("text"), f"{chunk_id} 第 {j} 段正文")
            heading = item.get("heading")
            if heading is not None and (not isinstance(heading, str) or
                                        not heading.strip() or "\n" in heading):
                raise DraftError(f"{chunk_id} 第 {j} 段标题无效")
            frame = None
            if item.get("frame_id") is not None:
                frame_id = item["frame_id"]
                if frame_id not in frame_by_id or frame_id in seen_frames:
                    raise DraftError(f"{chunk_id} 第 {j} 段图片 ID 不存在或重复")
                frame = frame_by_id[frame_id]
                lower = segs[a]["start"] if a == 0 else segs[a - 1]["end"]
                if not (lower <= frame["t"] <= segs[b]["end"]):
                    raise DraftError(f"{chunk_id} 第 {j} 段图片时间不属于该段")
                if not _inside(work, frame["path"]).is_file():
                    raise DraftError(f"图片文件丢失：{frame['path']}")
                seen_frames.add(frame_id)
            paragraphs.append({"heading": heading, "text": text,
                               "end": segs[b]["end"], "frame": frame})
            cursor = b + 1
        if cursor != len(segs):
            raise DraftError(f"{chunk_id} 末尾有 {len(segs) - cursor} 条字幕未覆盖")
        total += len(segs)
        previous_end = segs[-1]["end"]
    if total != manifest.get("segments"):
        raise DraftError("字幕数量与清单不符")
    return manifest, meta, paragraphs


def render_bundle(work: Path, force: bool = False, check: bool = False) -> dict:
    manifest, meta, paragraphs = validate_bundle(work)
    summary = {"chunks": len(manifest["chunks"]), "segments": manifest["segments"],
               "paragraphs": len(paragraphs),
               "images": sum(p["frame"] is not None for p in paragraphs)}
    if check:
        return summary
    work = Path(work)
    output = work / "course.md"
    if output.exists() and not force:
        raise DraftError(f"{output} 已存在；确认要替换时添加 --force")
    lines = [f"# {_text(meta['title'], '标题')}", "", _text(meta["intro"], "课程导语"), ""]
    assets = work / "assets"
    for i, para in enumerate(paragraphs, 1):
        if para["heading"]:
            lines.extend([f"## {para['heading'].strip()}", ""])
        lines.extend([para["text"], "", f"*{_time_link(para['end'], manifest['source'])}*", ""])
        if para["frame"]:
            assets.mkdir(exist_ok=True)
            source = _inside(work, para["frame"]["path"])
            name = f"work-p{i:03d}-{source.name}"
            shutil.copyfile(source, assets / name)
            lines.extend([f"![](assets/{name})", ""])
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return summary
