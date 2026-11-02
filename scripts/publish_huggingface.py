from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from amharic_news.labels import PUBLISHED_MODEL_ID, PUBLISHED_SPACE_ID


def _repo_info_or_none(api: HfApi, repo_id: str, repo_type: str):
    try:
        return api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        return None


def publish(
    release_dir: Path,
    model_id: str,
    space_id: str,
    token: str,
    space_hardware: str = "zero-a10g",
) -> None:
    if not token:
        raise ValueError("Set HF_TOKEN to a Project 04-specific Hugging Face write token.")
    api = HfApi(token=token)
    who = api.whoami()
    username = who.get("name") or who.get("fullname")
    if username != model_id.split("/", maxsplit=1)[0]:
        raise PermissionError(
            f"The supplied credential is for {username!r}, not the requested owner "
            f"{model_id.split('/', maxsplit=1)[0]!r}; nothing was uploaded."
        )
    model_folder = release_dir / "model"
    space_folder = release_dir / "space"
    if not model_folder.is_dir() or not space_folder.is_dir():
        raise FileNotFoundError(
            "Build a complete release first; model/ and space/ folders are required."
        )

    existing_model = _repo_info_or_none(api, model_id, "model")
    existing_space = _repo_info_or_none(api, space_id, "space")
    if existing_model or existing_space:
        existing = [
            repo_id
            for repo_id, info in ((model_id, existing_model), (space_id, existing_space))
            if info is not None
        ]
        raise FileExistsError(
            "Refusing to overwrite existing Hugging Face repositories: " + ", ".join(existing)
        )
    api.create_repo(repo_id=model_id, repo_type="model", private=False, exist_ok=False)
    model_commit = api.upload_folder(
        repo_id=model_id,
        repo_type="model",
        folder_path=str(model_folder),
        commit_message="Publish Project 04 Amharic news classifier",
    )
    print(f"Model published: https://huggingface.co/{model_id} ({model_commit.oid})")

    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            private=False,
            exist_ok=False,
            space_sdk="gradio",
            space_hardware=space_hardware,
        )
        space_commit = api.upload_folder(
            repo_id=space_id,
            repo_type="space",
            folder_path=str(space_folder),
            commit_message="Publish Project 04 Amharic news demo",
        )
        space_info = api.space_info(space_id)
        if space_info.private:
            raise RuntimeError("The Hugging Face Space is private; expected public visibility.")
        stage = space_info.runtime.stage if space_info.runtime else "unknown"
        print(
            f"Space published: https://huggingface.co/spaces/{space_id} "
            f"({space_commit.oid}); stage={stage}"
        )
    except HfHubHTTPError as exc:
        print(
            "The model is public, but Space creation/upload failed. "
            "Do not rerun the model upload; resolve the Hugging Face "
            "Space-account restriction first. "
            f"Hub response: {exc}"
        )
        raise


def publish_space_only(
    release_dir: Path,
    model_id: str,
    space_id: str,
    token: str,
    space_hardware: str,
) -> None:
    if not token:
        raise ValueError("Set HF_TOKEN to a Project 04-specific Hugging Face write token.")
    api = HfApi(token=token)
    who = api.whoami()
    username = who.get("name") or who.get("fullname")
    owner = space_id.split("/", maxsplit=1)[0]
    if username != owner:
        raise PermissionError(
            f"The supplied credential is for {username!r}, not the requested Space owner "
            f"{owner!r}; nothing was uploaded."
        )
    if _repo_info_or_none(api, model_id, "model") is None:
        raise FileNotFoundError(
            f"The already-published model {model_id!r} was not found; no Space was created."
        )
    space_folder = release_dir / "space"
    if not space_folder.is_dir():
        raise FileNotFoundError("Build the Space bundle first; release/space is required.")
    if _repo_info_or_none(api, space_id, "space") is not None:
        raise FileExistsError(f"Refusing to overwrite existing Hugging Face Space: {space_id}")

    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        private=False,
        exist_ok=False,
        space_sdk="gradio",
        space_hardware=space_hardware,
    )
    space_commit = api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=str(space_folder),
        commit_message="Publish Project 04 Amharic news ZeroGPU demo",
    )
    space_info = api.space_info(space_id)
    runtime = api.get_space_runtime(space_id)
    if space_info.private:
        raise RuntimeError("The Hugging Face Space is private; expected public visibility.")
    if runtime.requested_hardware != space_hardware and runtime.hardware != space_hardware:
        raise RuntimeError(
            f"Space was uploaded, but requested hardware {space_hardware!r} is not reflected by "
            f"the runtime (current={runtime.hardware!r}, requested={runtime.requested_hardware!r})."
        )
    print(
        f"Space published: https://huggingface.co/spaces/{space_id} ({space_commit.oid}); "
        f"stage={runtime.stage}; hardware={runtime.hardware}; "
        f"requested_hardware={runtime.requested_hardware}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish Project 04 to Hugging Face")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=PUBLISHED_MODEL_ID)
    parser.add_argument("--space-id", default=PUBLISHED_SPACE_ID)
    parser.add_argument("--space-only", action="store_true")
    parser.add_argument("--space-hardware", choices=("zero-a10g",), default="zero-a10g")
    args = parser.parse_args()
    try:
        token = os.environ.get("HF_TOKEN", "")
        if args.space_only:
            publish_space_only(
                args.release_dir,
                args.model_id,
                args.space_id,
                token,
                args.space_hardware,
            )
        else:
            publish(
                args.release_dir,
                args.model_id,
                args.space_id,
                token,
                args.space_hardware,
            )
    except (ValueError, PermissionError, FileExistsError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
