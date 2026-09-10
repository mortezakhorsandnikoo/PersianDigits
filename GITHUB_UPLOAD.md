# How to publish this release on GitHub

You already have a repo at **https://github.com/mortezakhorsandnikoo/PersianDigits**
(currently holding `persianDigits.rar`). Publishing these files there keeps the same
URL your paper already cites — recommended. Everything here is small (the whole release
is well under GitHub's 100 MB/file limit), so you do **not** need Git LFS.

## Before you upload

Run the notebook (`persian_digits_rigorous_cnn.ipynb`) top to bottom once. Its last two
sections create, inside this `PersianDigits_release` folder:
  - `PersianDigits.npz`, `PersianDigits.csv`  (dataset)
  - `model/persian_digit_cnn.pt`, `model/preprocessing.json`  (trained model)
Then copy your `dataset/0..9` image folders into this folder too.

## Option A — Git command line (recommended)

```bash
cd PersianDigits_release

git init
git remote add origin https://github.com/mortezakhorsandnikoo/PersianDigits.git
git fetch origin
git checkout -b main

git add .
git commit -m "Add Persian digit dataset (NPZ/CSV + images), trained CNN weights, loader, and reproducible notebook"

# Push. This keeps the old .rar too; if you'd rather replace everything, see the note below.
git pull origin main --allow-unrelated-histories   # merge with the existing repo
git push -u origin main
```

If you prefer to **replace** the old contents entirely (drop the `.rar`):

```bash
git push -u origin main --force
```

Use `--force` only if you're sure you want to overwrite what's currently in the repo.

## Option B — GitHub website (no command line)

1. Go to https://github.com/mortezakhorsandnikoo/PersianDigits
2. Click **Add file → Upload files**.
3. Drag in everything from this folder (including the `dataset/` and `model/` folders).
   You may need to upload the `dataset` subfolders in a few batches if there are many files.
4. Write a commit message and click **Commit changes**.

## Making the license official (optional but tidy)

The included `LICENSE` names CC BY 4.0 and links to the full legal text. If you'd like the
complete license text embedded by GitHub itself: on the repo page click **Add file →
Create new file**, name it `LICENSE`, click **Choose a license template**, pick
*Creative Commons Attribution 4.0 International*, and commit — this overwrites the stub
with the official full text.

## After uploading

- Update the manuscript's "Availability of data and material" line to point to the repo.
- Consider creating a **Release** (Releases → Draft a new release, tag e.g. `v1.0`) and, if
  you want a citable DOI, connect the repo to Zenodo so each release mints one.
