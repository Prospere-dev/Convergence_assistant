# Convergence Assistant

**Analyse de suites, séries et séries entières, avec démonstrations rédigées en LaTeX.**

Convergence Assistant est une application web locale (Flask + SymPy + MathJax). Tu saisis une expression, l'application choisit les critères adaptés, rédige la démonstration étape par étape et trace les graphes correspondants.

*English summary: a local Flask/SymPy web app that studies the convergence of sequences, series, function sequences/series and power series, and writes the reasoning in LaTeX (rendered with MathJax) with matplotlib plots.*

<p align="center"><img src="docs/images/demo_curv_green_disque.gif" width="760" alt="Démonstration : Green–Riemann sur le disque unité"></p>

*Démonstration : théorème de Green–Riemann sur le disque unité, calcul du bord et de l'intégrale double.*


## Ce que l'application traite

| Type | Entrée | Sortie |
|---|---|---|
| Suite numérique | terme général `u_n` | limite, verdict, graphe |
| Série numérique | `u_n` et indice de départ | critères de convergence appliqués, démonstration, graphes de `u_n` et des sommes partielles |
| Suite de fonctions | `f_n(x)` et intervalle | étude de la limite, graphes de la famille et de la limite |
| Série de fonctions | `u_n(x)`, indice de départ, intervalle | étude de convergence, sommes partielles tracées |
| Série entière | `a_n x^n`, indice de départ, intervalle | rayon de convergence (Cauchy-Hadamard, d'Alembert), étude sur l'intervalle, fonction somme |
| Développement en série entière | `f(x)`, ordre | série, rayon, polynômes de Taylor tracés |
| Série de Fourier | `f(x)` périodique, intervalle | coefficients, convergence, sommes partielles tracées |
| Intégrale (propre, impropre, Lebesgue) | `f(x)`, intervalle | points à problème, intégrabilité, calcul en chaîne d'égalités alignées |
| Convergence monotone / dominée / Fatou | `f_n(x)`, intervalle | limite, domination, passage à la limite sous l'intégrale |
| Intégrale multiple | `f`, domaine par inégalités | Fubini / Fubini–Tonelli, changement de variables, graphes |
| Intégrale curviligne et Green–Riemann | champ `P, Q` (ou `P, Q, R`), arc `x=..., y=..., t in [a,b]` ou domaine `D` | circulation, potentiel, aire, vérification de Green–Riemann |

### Critères implémentés pour les séries numériques

Divergence grossière, Riemann, Bertrand, séries polylogarithmiques, d'Alembert, Cauchy (racine), Leibniz (séries alternées, positivité et décroissance vérifiées), comparaison, équivalents, convergence absolue.

## Installation

Prérequis : Python 3.10 ou plus récent.

```bash
git clone https://github.com/Prospere-dev/convergence-assistant.git
cd convergence-assistant
python -m venv .venv
# Windows : .venv\Scripts\activate      Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Ouvre ensuite <http://127.0.0.1:5000>. MathJax est chargé depuis un CDN : une connexion internet est nécessaire pour l'affichage des formules.

## Exemples à essayer

| Type | Saisie | Ce que tu dois observer |
|---|---|---|
| Série | `1/n^2`, indice 1 | convergence (Riemann) |
| Série | `1/n`, indice 1 | divergence |
| Série | `(-1)^n/n`, indice 1 | convergence par Leibniz |
| Série entière | `x^n/n!`, indice 0 | rayon infini |
| Série entière | `(-1)^n*x^n`, indice 0 | rayon 1 |
| Développement | `exp(x)`, ordre 10 | polynômes de Taylor tracés avec la fonction |
| Suite de fonctions | `x^n/(1+n)` sur `[0, 1]` | limite et graphes |
| Intégrale (Lebesgue) | `sin(x)/x` sur `]0, +oo[` | convergence conditionnelle, f non intégrable |
| Intégrale multiple | `x*y`, domaine `0<=x<=1, x<=y<=2*x` | Fubini, chaîne d'égalités |
| Intégrale curviligne | champ `-y, x`, domaine `x^2+y^2<=1` | Green–Riemann : 2π des deux côtés |
| Intégrale curviligne | champ `2*x*y, x^2`, arc `x=cos(t), y=sin(t), t in [0,2*pi]` | résultat 0 prévu (champ de gradient) |

Notation : `^` pour les puissances, `n!` pour la factorielle, `oo` pour +∞ et `-oo` pour −∞ dans les bornes de l'intervalle. Les crochets inversés `] a , b [` désignent un intervalle ouvert.

## Architecture

```
app.py              cœur mathématique (SymPy) et route Flask /compute
lebesgue.py         intégrales propres, impropres, Lebesgue ; théorèmes de convergence
multi.py            intégrales multiples (Fubini, changements de variables)
fourier.py          séries de Fourier
curv.py             intégrales curvilignes et Green–Riemann
templates/index.html  formulaire et zone de résultats
static/app.js         interface : envoi JSON, affichage des messages
static/style.css      mise en forme
test_app.py           tests pytest
```

Flux : le navigateur envoie un JSON (`type_input`, `user_input1`, `user_input2`, bornes de l'intervalle) à `/compute`. Le serveur renvoie une liste de messages typés (`latex`, `info`, `error`, `plot`). Le JavaScript les affiche, MathJax rend le LaTeX, les graphes arrivent en PNG encodé en base64.

## Galerie

Captures prises sur l'application en fonctionnement (formules rendues par MathJax, graphes produits par matplotlib).

### Démonstrations animées

Défilement complet d'une étude, de la saisie à la conclusion.

<p align="center"><img src="docs/images/demo_serie_alternee.gif" width="760" alt="Série alternée"></p>

*Série ∑(−1)ⁿ/n : critère de Leibniz.*

<p align="center"><img src="docs/images/demo_serie_entiere_ln.gif" width="760" alt="Série entière de ln(1+x)"></p>

*Série entière ∑(−1)ⁿ⁺¹xⁿ/n sur [−2, 2] : convergence, divergence, étude des bords.*

<p align="center"><img src="docs/images/demo_integrale_divergente.gif" width="760" alt="Intégrale divergente"></p>

*∫₁^{+∞} dx/x : intégrale impropre divergente.*

### Suites numériques

La limite est calculée puis la nature de la suite est déduite.

**uₙ = (1 + 1/n)ⁿ : convergence vers e**

<p align="center"><img src="docs/images/an_suite_e_1.png" width="760" alt="Suite (1+1/n)^n (1)"></p>

<p align="center"><img src="docs/images/an_suite_e_2.png" width="760" alt="Suite (1+1/n)^n (2)"></p>

**uₙ = n²/(n+1) : divergence, limite +∞**

<p align="center"><img src="docs/images/an_suite_div_1.png" width="760" alt="Suite n^2/(n+1) (1)"></p>

<p align="center"><img src="docs/images/an_suite_div_2.png" width="760" alt="Suite n^2/(n+1) (2)"></p>

### Séries numériques

Chaque conclusion cite le critère employé et ses hypothèses.

**∑ 1/n² : convergence (critère de Riemann)**

<p align="center"><img src="docs/images/an_serie_riemann_1.png" width="760" alt="Série 1/n^2 (1)"></p>

<p align="center"><img src="docs/images/an_serie_riemann_2.png" width="760" alt="Série 1/n^2 (2)"></p>

<p align="center"><img src="docs/images/an_serie_riemann_3.png" width="760" alt="Série 1/n^2 (3)"></p>

**∑ 1/n : divergence (série harmonique)**

<p align="center"><img src="docs/images/an_serie_harmo_1.png" width="760" alt="Série harmonique (1)"></p>

<p align="center"><img src="docs/images/an_serie_harmo_2.png" width="760" alt="Série harmonique (2)"></p>

<p align="center"><img src="docs/images/an_serie_harmo_3.png" width="760" alt="Série harmonique (3)"></p>

**∑ (−1)ⁿ/n : convergence (critère de Leibniz)**

<p align="center"><img src="docs/images/an_serie_alt_1.png" width="760" alt="Série alternée (1)"></p>

<p align="center"><img src="docs/images/an_serie_alt_2.png" width="760" alt="Série alternée (2)"></p>

<p align="center"><img src="docs/images/an_serie_alt_3.png" width="760" alt="Série alternée (3)"></p>

**∑ 1/(n ln²n) : convergence (critère des logarithmes itérés)**

<p align="center"><img src="docs/images/an_serie_bertrand_1.png" width="760" alt="Série de Bertrand (1)"></p>

<p align="center"><img src="docs/images/an_serie_bertrand_2.png" width="760" alt="Série de Bertrand (2)"></p>

<p align="center"><img src="docs/images/an_serie_bertrand_3.png" width="760" alt="Série de Bertrand (3)"></p>

### Suites et séries de fonctions

Convergence simple, convergence uniforme, conséquences sur la limite ou la somme, avec les graphes.

**fₙ(x) = xⁿ sur [0, 1] : convergence simple, pas de convergence uniforme**

<p align="center"><img src="docs/images/an_sf_xn_1.png" width="760" alt="Suite de fonctions x^n (1)"></p>

<p align="center"><img src="docs/images/an_sf_xn_2.png" width="760" alt="Suite de fonctions x^n (2)"></p>

**∑ xⁿ/n² sur [−1, 1] : convergence uniforme, somme continue, intégration terme à terme**

<p align="center"><img src="docs/images/an_srf_xn2_1.png" width="760" alt="Série de fonctions x^n/n^2 (1)"></p>

<p align="center"><img src="docs/images/an_srf_xn2_2.png" width="760" alt="Série de fonctions x^n/n^2 (2)"></p>

### Séries entières

**∑ xⁿ/n! sur [−2, 2] : convergence absolue, somme eˣ**

<p align="center"><img src="docs/images/an_se_exp_1.png" width="760" alt="Série entière exponentielle (1)"></p>

<p align="center"><img src="docs/images/an_se_exp_2.png" width="760" alt="Série entière exponentielle (2)"></p>

**∑ (−1)ⁿ⁺¹xⁿ/n sur [−2, 2] : rayon 1, étude des bords, somme ln(1+x) et sommes partielles**

<p align="center"><img src="docs/images/an_se_ln_1.png" width="760" alt="Série entière de ln(1+x) (1)"></p>

<p align="center"><img src="docs/images/an_se_ln_2.png" width="760" alt="Série entière de ln(1+x) (2)"></p>

### Intégrales de Lebesgue, propres et impropres

Chaque calcul d'intégrale s'écrit en chaîne d'égalités alignées, chacune avec sa justification brève. L'application explique aussi ce que signifie « intégrable » (convergence absolue) et pourquoi une intégrale impropre peut converger sans l'être.

**∫₀^{+∞} x e^{-x} dx : intégration par parties**

<p align="center"><img src="docs/images/leb_x_exp_1.png" width="760" alt="leb x exp 1"></p>

<p align="center"><img src="docs/images/leb_x_exp_2.png" width="760" alt="leb x exp 2"></p>

<p align="center"><img src="docs/images/leb_x_exp_3.png" width="760" alt="leb x exp 3"></p>

**∫_ℝ e^{-x²} dx : lemme de Gauss**

<p align="center"><img src="docs/images/leb_gauss_1.png" width="760" alt="leb gauss 1"></p>

<p align="center"><img src="docs/images/leb_gauss_2.png" width="760" alt="leb gauss 2"></p>

<p align="center"><img src="docs/images/leb_gauss_3.png" width="760" alt="leb gauss 3"></p>

**∫₀^{+∞} sin x / x dx : convergence conditionnelle, f ∉ L¹**

<p align="center"><img src="docs/images/leb_dirichlet_1.png" width="760" alt="leb dirichlet 1"></p>

<p align="center"><img src="docs/images/leb_dirichlet_2.png" width="760" alt="leb dirichlet 2"></p>

<p align="center"><img src="docs/images/leb_dirichlet_3.png" width="760" alt="leb dirichlet 3"></p>

**∫₀¹ dx/√x : borne à problème, intégrale impropre convergente**

<p align="center"><img src="docs/images/leb_sqrt_1.png" width="760" alt="leb sqrt 1"></p>

<p align="center"><img src="docs/images/leb_sqrt_2.png" width="760" alt="leb sqrt 2"></p>

<p align="center"><img src="docs/images/leb_sqrt_3.png" width="760" alt="leb sqrt 3"></p>

**∫₁^{+∞} dx/x : intégrale impropre divergente, f ∉ L¹**

<p align="center"><img src="docs/images/an_int_div_1.png" width="760" alt="Intégrale de 1/x (1)"></p>

<p align="center"><img src="docs/images/an_int_div_2.png" width="760" alt="Intégrale de 1/x (2)"></p>

<p align="center"><img src="docs/images/an_int_div_3.png" width="760" alt="Intégrale de 1/x (3)"></p>

**∫₀¹ ln x dx = −1 : borne singulière, contrôle numérique**

<p align="center"><img src="docs/images/an_int_log_1.png" width="760" alt="Intégrale de ln x (1)"></p>

<p align="center"><img src="docs/images/an_int_log_2.png" width="760" alt="Intégrale de ln x (2)"></p>

<p align="center"><img src="docs/images/an_int_log_3.png" width="760" alt="Intégrale de ln x (3)"></p>

**∫_ℝ dx/(1+x²) : convergence, intégrales tronquées vers π/2 de chaque côté**

<p align="center"><img src="docs/images/an_int_arctan_1.png" width="760" alt="Intégrale de 1/(1+x^2) (1)"></p>

<p align="center"><img src="docs/images/an_int_arctan_2.png" width="760" alt="Intégrale de 1/(1+x^2) (2)"></p>

<p align="center"><img src="docs/images/an_int_arctan_3.png" width="760" alt="Intégrale de 1/(1+x^2) (3)"></p>

### Séries de Fourier



**f(x) = |x| sur [−π, π] : coefficients, convergence, graphes des sommes partielles**

<p align="center"><img src="docs/images/fourier_abs_1.png" width="760" alt="fourier abs 1"></p>

<p align="center"><img src="docs/images/fourier_abs_2.png" width="760" alt="fourier abs 2"></p>

<p align="center"><img src="docs/images/fourier_abs_3.png" width="760" alt="fourier abs 3"></p>

<p align="center"><img src="docs/images/fourier_abs_4.png" width="760" alt="fourier abs 4"></p>

### Théorèmes de convergence pour les intégrales



**f_n(x) = n x e^{-n x²} sur [0, 1] : convergence monotone, dominée, Fatou**

<p align="center"><img src="docs/images/conv_nxexp_1.png" width="760" alt="conv nxexp 1"></p>

<p align="center"><img src="docs/images/conv_nxexp_2.png" width="760" alt="conv nxexp 2"></p>

### Intégrales multiples (Fubini, Fubini–Tonelli, changements de variables)



**∬_{ℝ²} e^{-x²-y²} : passage en polaires**

<p align="center"><img src="docs/images/multi_gauss_1.png" width="760" alt="multi gauss 1"></p>

<p align="center"><img src="docs/images/multi_gauss_2.png" width="760" alt="multi gauss 2"></p>

**∬ x y sur le domaine 0 ≤ x ≤ 1, x ≤ y ≤ 2x : Fubini**

<p align="center"><img src="docs/images/multi_triangle_1.png" width="760" alt="multi triangle 1"></p>

<p align="center"><img src="docs/images/multi_triangle_2.png" width="760" alt="multi triangle 2"></p>

### Intégrales curvilignes et théorème de Green–Riemann

Le champ de vecteurs se note V = (P, Q). La formule ∮ V·dℓ = ∮ P dx + Q dy = ∫ (P x′ + Q y′) dt = ∬ (∂Q/∂x − ∂P/∂y) dx dy s'affiche en une ligne, puis chaque morceau du bord est paramétré et calculé. Quand le résultat est prévisible (contour fermé, champ de gradient, rotationnel nul), il est annoncé avec sa justification, sans calcul.

**V = (−y, x) sur le disque unité : circulation = 2π des deux côtés**

<p align="center"><img src="docs/images/curv_green_disque_1.png" width="760" alt="curv green disque 1"></p>

<p align="center"><img src="docs/images/curv_green_disque_2.png" width="760" alt="curv green disque 2"></p>

**aire de l'ellipse x²/4 + y²/9 ≤ 1 par Green–Riemann : 6π**

<p align="center"><img src="docs/images/curv_aire_ellipse_1.png" width="760" alt="curv aire ellipse 1"></p>

<p align="center"><img src="docs/images/curv_aire_ellipse_2.png" width="760" alt="curv aire ellipse 2"></p>

**V = (y², x) sur le triangle x ≥ 0, y ≥ 0, x + y ≤ 1 : trois morceaux de bord, relation de Chasles**

<p align="center"><img src="docs/images/curv_green_triangle_1.png" width="760" alt="curv green triangle 1"></p>

<p align="center"><img src="docs/images/curv_green_triangle_2.png" width="760" alt="curv green triangle 2"></p>

**V = (2xy, x²) sur le cercle : résultat 0 annoncé par le lemme de Poincaré**

<p align="center"><img src="docs/images/curv_prevu_gradient_1.png" width="760" alt="curv prevu gradient 1"></p>

<p align="center"><img src="docs/images/curv_prevu_gradient_2.png" width="760" alt="curv prevu gradient 2"></p>

**V = (−y, x)/(x²+y²) sur le cercle : rotationnel nul mais circulation 2π, pas de potentiel sur ℝ²∖{0}**

<p align="center"><img src="docs/images/curv_singulier_1.png" width="760" alt="curv singulier 1"></p>

<p align="center"><img src="docs/images/curv_singulier_2.png" width="760" alt="curv singulier 2"></p>

**V = (yz, xz, xy) le long d'une hélice : champ de gradient en 3D, contrôle par le potentiel**

<p align="center"><img src="docs/images/curv_helice_1.png" width="760" alt="curv helice 1"></p>

<p align="center"><img src="docs/images/curv_helice_2.png" width="760" alt="curv helice 2"></p>

**longueur de l'arc y = x² sur [0, 1]**

<p align="center"><img src="docs/images/curv_longueur_parabole_1.png" width="760" alt="curv longueur parabole 1"></p>

<p align="center"><img src="docs/images/curv_longueur_parabole_2.png" width="760" alt="curv longueur parabole 2"></p>

## Tests

```bash
pip install pytest
python -m pytest -v
```

## Limites connues

- Le calcul est symbolique (SymPy). Sur des expressions exotiques, SymPy peut échouer ou ne pas conclure. L'application doit alors le signaler, sans affirmer un verdict.
- Les critères sont appliqués dans un ordre fixé. Un résultat « pas de conclusion » ne prouve rien sur la nature de la série.
- Usage local uniquement. Les expressions saisies sont interprétées par `sympify`, ce qui permet d'exécuter du code Python. Ne déploie pas l'application sur un serveur public sans restreindre l'analyse des entrées et sans désactiver `debug=True`.
- Les graphes montrent un comportement numérique sur un nombre fini de termes. Ils illustrent un résultat, ils ne le démontrent pas.

## Pistes d'amélioration

- Séparer `app.py` en modules (critères, séries de fonctions, séries entières, routes).
- Remplacer l'état global de sortie par un objet créé à chaque requête.
- Analyse sécurisée des entrées (`parse_expr` avec noms autorisés).
- Export de la démonstration en `.tex` ou PDF.

## Licence

Distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE).
