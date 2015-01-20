(TeX-add-style-hook
 "paper"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("oasics" "a4paper" "UKenglish")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8")))
   (TeX-run-style-hooks
    "latex2e"
    "oasics"
    "oasics10"
    "microtype"
    "graphicx"
    "booktabs"
    "multirow"
    "color"
    "colortbl"
    "gb4e"
    "inputenc")
   (LaTeX-add-labels
    "sec:previous-work"
    "sec:data"
    "sec:models"
    "sec:results"
    "tab:results-all"
    "tab:results-noun"
    "fig:prec-recall-curve")
   (LaTeX-add-bibliographies)))

