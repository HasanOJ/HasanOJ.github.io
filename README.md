# HasanOJ.github.io — Usage Guide

This repo contains a Hugo site with helper scripts to prepare markdown posts with citations and images.

## Quick Start

- Install Hugo (extended): https://gohugo.io/getting-started/installing/
- Ensure Python 3 is available: `python --version`
- Clone repo and open in VS Code.

## Add a New Post

- Write your draft anywhere (e.g., Obsidian). Save as a `.md` file.
- Place images/screenshots in the configured attachments folder:
	- `ATTACHMENTS_DIR` (see `scripts/scripts.txt`): `C:\Users\ASUS\Documents\Obsidian Vault\Screenshots`

## Process the Post (normalize citations/images)

Use the main script `scripts/process_post.py` to convert any markdown file into the proper Hugo post structure.

```pwsh
python scripts/process_post.py "path\to\your-post.md"
```

test with:
```pwsh
python scripts/process_post.py "C:\Users\ASUS\Documents\Obsidian Vault\Blog\Generative Models (unedited).md"
```

What the script does:
- Moves the file to `content/posts/<slug>/index.md`
- Copies referenced images to `content/posts/<slug>/images/`
- Converts citations `[@key]` → clickable numeric references `[1]`
- Converts Obsidian-style images `![[image.png|Caption]]` → HTML figures
- Adds minimal front matter if missing

Config references:
- `ATTACHMENTS_DIR`, `POSTS_DIR`, `BIB_FILE` detailed in `scripts/scripts.txt`
- BibTeX file expected at repo root: `references.bib`

## Run Locally (preview before publishing)

From the repo root, start Hugo server:

```pwsh
hugo server -D
```

- Visit `http://localhost:1313/` to preview.
- `-D` includes draft posts; remove it to view only published.

## Publish

- Build static site:

```pwsh
hugo
```

- Deploy the contents of `public/` (GitHub Pages typical setup pushes from default branch or `docs/` depending on your workflow).

## Notes

- Alternative citation styles are archived under `scripts/archive/`. This is deprecated.
- If a citation key is missing, add it to `references.bib`.
