from pathlib import Path
from urllib.parse import urljoin
import tomllib

from bs4 import BeautifulSoup
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def download_from_gdrive(file_id: str, dest: Path) -> None:
    initial_url = "https://docs.google.com/uc"
    with requests.Session() as s:
        r = s.get(
            initial_url,
            params={"id": file_id, "export": "download"},
            stream=True,
        )
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", "").lower():
            soup = BeautifulSoup(r.text, "html.parser")
            form = soup.find("form", id="download-form") or soup.find("form")
            if form is None:
                raise RuntimeError(
                    f"Google Drive returned HTML with no download form for file id "
                    f"{file_id!r}. The file may be private, deleted, or rate-limited."
                )
            action_url = form.get("action")
            if not action_url:
                raise RuntimeError("Drive confirmation form has no action URL.")
            action_url = urljoin(r.url, action_url)
            params = {
                inp.get("name"): inp.get("value", "")
                for inp in form.find_all("input", attrs={"type": "hidden"})
                if inp.get("name")
            }
            r = s.get(action_url, params=params, stream=True)
            r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(1 << 15):
                if chunk:
                    f.write(chunk)


def ensure_file(file_id: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[prepare] {dest.relative_to(PROJECT_ROOT)} already present, skipping.")
        return dest
    if file_id.startswith("REPLACE_WITH_"):
        raise RuntimeError(
            f"GDrive file ID for {dest.name} is still the placeholder "
            f"'{file_id}'. Edit pyproject.toml [[tool.prepare.downloads]] to set the real ID."
        )
    print(f"[prepare] Downloading {dest.relative_to(PROJECT_ROOT)} from Google Drive...")
    download_from_gdrive(file_id, dest)
    print(f"[prepare] Saved {dest.relative_to(PROJECT_ROOT)}.")
    return dest


def main() -> None:
    with PYPROJECT.open("rb") as f:
        cfg = tomllib.load(f)
    downloads = cfg["tool"]["prepare"]["downloads"]
    for item in downloads:
        ensure_file(item["gdrive_id"], PROJECT_ROOT / item["path"])


if __name__ == "__main__":
    main()
