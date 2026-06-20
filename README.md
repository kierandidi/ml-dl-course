# Machine Learning & Deep Learning — Two-Week Intensive Course

**Instructor:** Kieran Didi  
**Dates:** 17–28 August 2026  
**Website:** [ml-dl-course.netlify.app](https://ml-dl-course.netlify.app/) *(set after first deploy — see [DEPLOY.md](DEPLOY.md))*

Ten-day course covering classical ML/DL (Week 1) and generative modeling — diffusion, flows, and autoregressive LLMs (Week 2). Each day includes lecture notes, Typst slides, and a Jupyter practical.

## Repository layout

| Path | Contents |
|------|----------|
| `lectures/_posts/` | Daily lecture pages (Jekyll) |
| `_projects/` | Practical and assessment project pages |
| `slides/days/` | Typst slide sources (`day01.typ` … `day10.typ`) |
| `assets/slides/` | Compiled PDF slides (generated) |
| `notebooks/practicals/` | Daily practical notebooks |
| `notebooks/assessment/` | Final assessment notebook |
| `notebooks/data/` | Synthetic datasets |
| `scripts/` | Slide and lecture generation scripts |

---

## Build the course website (Jekyll + Hydejack)

### Prerequisites

- **Ruby** 3.2+ and **Bundler**
- **Node.js** (required by `kramdown-math-katex` for KaTeX math)
- **Java** runtime (optional fallback for ExecJS)

### Local setup

```bash
cd ml-dl-course
export GEM_HOME="$(pwd)/vendor/ruby"
export PATH="$(pwd)/vendor/ruby/bin:$PATH"
gem install bundler -v 2.4.22 --install-dir "$GEM_HOME" --bindir "$GEM_HOME/bin"
bundle config set --local path vendor/bundle
bundle install
```

Install Typst for slides: `brew install typst` or download from [typst.app](https://typst.app/).

### Serve locally

```bash
bundle exec jekyll serve --livereload
```

Open [http://localhost:4000](http://localhost:4000). Search is disabled in development by default; set `JEKYLL_ENV=production` to enable it:

```bash
JEKYLL_ENV=production bundle exec jekyll serve
```

### Production build

```bash
JEKYLL_ENV=production bundle exec jekyll build
# Output in _site/
```

---

## Compile slides (Typst)

Slides live in `slides/days/*.typ`. Install [Typst](https://typst.app/) (≥ 0.11), then compile each day:

```bash
typst compile slides/days/day01.typ assets/slides/day01.pdf
# … repeat for day02 … day10
```

Or use the Python helpers (requires Typst on `PATH`):

```bash
python3 scripts/extract_figures.py    # pull PNGs from UCL/MIT/PPTX/PDF sources
python3 scripts/generate_slides.py
bash scripts/build_slides.sh
python3 scripts/generate_project_pages.py  # Exercises hub (projects.md) + final assessment
python3 scripts/setup_lesson_heroes.py     # cropped concept hero image per day
```

PDFs are served from `/assets/slides/dayNN.pdf` on the site.

---

## Notebooks

### Environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r notebooks/requirements-notebooks.txt
jupyter notebook notebooks/
```

Open practicals from `notebooks/practicals/dayNN.ipynb` or the assessment at `notebooks/assessment/final_assessment.ipynb`.

> **Note:** The `notebooks/` directory is excluded from the Jekyll build (`_config.yml`). Clone the repo to run notebooks; project pages link to paths under `notebooks/`.

---

## Deploy to Netlify

The repo includes [`netlify.toml`](netlify.toml):

```toml
[build]
  command = "bundle exec jekyll build"
  publish = "_site"

[build.environment]
  JEKYLL_ENV = "production"
  RUBY_VERSION = "3.2.2"
```

1. Connect the Git repository in the Netlify dashboard.
2. Set **Build command** to `bundle exec jekyll build` and **Publish directory** to `_site` (defaults from `netlify.toml`).
3. Ensure Ruby 3.2.2 and Node.js are available (Netlify Ruby build image includes Node).
4. Optionally add a **pre-build** step to compile Typst slides if PDFs are not committed:

   ```bash
   python scripts/generate_slides.py
   bundle exec jekyll build
   ```

Push to the default branch; Netlify rebuilds automatically.

---

## Regenerate lecture content

Lecture markdown is produced by scripts in `scripts/`:

```bash
python scripts/generate_lectures.py      # all days 1–10 (content_dayNN.py modules)
```

Review diffs before committing generated posts.

---

## License

See [LICENSE.md](LICENSE.md) and [NOTICE.md](NOTICE.md). Course materials © 2026 Kieran Didi.
