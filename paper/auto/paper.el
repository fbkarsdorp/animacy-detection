(TeX-add-style-hook
 "paper"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper" "twocolumn" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "graphicx"
    "covington"
    "times"
    "booktabs"
    "multirow"
    "color"
    "colortbl"
    "inputenc")
   (LaTeX-add-labels
    "tab:results-all"
    "tab:results-noun")
   (LaTeX-add-bibliographies)))

