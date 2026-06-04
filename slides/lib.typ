#import "@preview/touying:0.7.3": *
#import themes.metropolis: *

#let course-primary = rgb("#4FB1BA")
#let course-dark = rgb("#193747")

#let course-theme(title: none, subtitle: none, body) = {
  show: metropolis-theme.with(
    aspect-ratio: "16-9",
    config-colors(primary: course-primary, secondary: course-dark),
    config-info(
      title: title,
      subtitle: subtitle,
      author: [Kieran Didi],
      date: datetime(year: 2026, month: 8, day: 17),
      institution: [ML & Deep Learning Course],
    ),
  )
  set text(size: 20pt)
  body
}

#let KL = [$D_"KL"$]
#let fig(path, caption: none) = {
  figure(
    image(path, width: 92%),
    caption: caption,
  )
}

#let section-slide(title) = [
  #set align(center + horizon)
  #text(size: 36pt, weight: "bold", fill: course-primary)[#title]
]
