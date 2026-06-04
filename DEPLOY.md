# Deploying the course website (GitHub + Netlify)

This site uses **Jekyll + Hydejack + KaTeX**, same as [sde-course](https://github.com/kierandidi/sde-course) and [bioinformatics-course-hd](https://github.com/kierandidi/bioinformatics-course-hd). **Netlify** builds the site on every push (GitHub Pages’ default builder does not run KaTeX correctly).

## 1. Create a new GitHub repository

Do **not** push this folder to `sde-course` — it was copied from that template and may still have the wrong `git remote`.

```bash
cd /Users/kdidi/Downloads/ml_course_august/ml-dl-course

# If remote still points at sde-course, re-init (only if you have not pushed ml-dl content yet):
# rm -rf .git && git init && git branch -M main

git init   # skip if you already have a clean repo for this course only
git add .
git commit -m "Initial ML & DL two-week course site"
```

On GitHub: **New repository** → e.g. `ml-dl-course` (public) → no README/license (you already have them).

```bash
git remote add origin git@github.com:kierandidi/ml-dl-course.git
git push -u origin main
```

## 2. Connect Netlify (automatic builds)

1. Log in at [netlify.com](https://www.netlify.com/) → **Add new site** → **Import an existing project** → **GitHub** → select `ml-dl-course`.
2. Build settings (should match [`netlify.toml`](netlify.toml)):

   | Setting | Value |
   |---------|--------|
   | Build command | `bundle exec jekyll build` |
   | Publish directory | `_site` |
   | Ruby version | `3.2.2` (env var `RUBY_VERSION`) |

3. Deploy. Netlify runs `bundle install` + `jekyll build` on each push to `main`.
4. Optional: **Site settings → Domain management** → custom subdomain e.g. `ml-dl-course.netlify.app`.

## 3. Set the site URL in Jekyll

After you know the Netlify URL, uncomment and set in [`_config.yml`](_config.yml):

```yaml
url: https://ml-dl-course.netlify.app
```

Then commit and push so feeds, sitemap, and absolute links resolve correctly.

## 4. What gets published automatically

| Path in repo | On the live site |
|--------------|------------------|
| `lectures/_posts/*.md` | **Lessons** (blog) — notes, TOC, KaTeX |
| `assets/slides/dayNN.pdf` | Linked as **Slides** on each day |
| `_projects/dayNN-practical.md` | **Practicals** — derivations + notebook link |
| `assets/img/lessons/dayNN.png` | **Title/hero image** on each lesson card |
| `notebooks/` | Not served by Jekyll (excluded); link from practical pages |

Slides and practicals are linked at the top of every lesson post:

```markdown
### [Slides](/assets/slides/day01.pdf)
### [Practical](/projects/day01-practical/)
```

## 5. Local preview (before push)

```bash
export GEM_HOME="$(pwd)/vendor/ruby"
export PATH="$(pwd)/vendor/ruby/bin:$PATH"
bundle install
JEKYLL_ENV=production bundle exec jekyll serve
```

Open http://localhost:4000

## 6. Rebuild slides after editing Typst

```bash
python3 scripts/extract_figures.py      # optional: refresh PNGs
python3 scripts/generate_slides.py
bash scripts/build_slides.sh
git add assets/slides assets/img/lessons
git commit -m "Update slides and lesson heroes"
git push
```

## 7. GitHub Actions (optional alternative to Netlify)

If you prefer build on GitHub instead of Netlify, add `.github/workflows/jekyll.yml` that runs `bundle exec jekyll build` and uploads `_site` to GitHub Pages. Netlify is still recommended because it matches your other courses and handles Ruby/KaTeX without extra workflow maintenance.
