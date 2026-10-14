# Convergence Assistant

Application web qui étudie la convergence des suites, des séries, des séries entières, des suites et séries de fonctions, des séries de Fourier et des intégrales (une, plusieurs variables, curvilignes). On écrit une expression, l'application rédige la solution étape par étape, avec les formules en LaTeX et les graphes. Elle tourne en local avec Flask, SymPy, matplotlib et MathJax.

<p align="center"><img src="docs/img/demo_curv_green_disque.gif" width="760" alt="Green–Riemann sur le disque unité, de la saisie à la conclusion"></p>

## Exemples

Chaque exemple montre la saisie, toutes les étapes du calcul telles que l'application les écrit, puis une explication des graphes. Les captures viennent de l'application en fonctionnement.

### Théorème de Green–Riemann sur le disque unité, champ V = (−y, x)

**Saisie :** type « Intégrale curviligne / Green–Riemann », champ `-y, x`, domaine `x^2+y^2<=1`.

<p align="center"><img src="docs/img/curv_green_disque_1.png" width="760" alt="Théorème de Green–Riemann sur le disque unité, champ V = (−y, x), étape 1"></p>

<p align="center"><img src="docs/img/curv_green_disque_2.png" width="760" alt="Théorème de Green–Riemann sur le disque unité, champ V = (−y, x), étape 2"></p>

<p align="center"><img src="docs/img/curv_green_disque_3.png" width="760" alt="Théorème de Green–Riemann sur le disque unité, champ V = (−y, x), étape 3"></p>

<p align="center"><img src="docs/img/curv_green_disque_4.png" width="760" alt="Théorème de Green–Riemann sur le disque unité, champ V = (−y, x), étape 4"></p>

<p align="center"><img src="docs/img/curv_green_disque_5.png" width="760" alt="Théorème de Green–Riemann sur le disque unité, champ V = (−y, x), étape 5"></p>

<p align="center"><img src="docs/img/curv_green_disque_6.png" width="760" alt="Théorème de Green–Riemann sur le disque unité, champ V = (−y, x), étape 6"></p>

**Résultat :** La circulation de V le long du cercle vaut 2π. L'intégrale double de ∂Q/∂x − ∂P/∂y, qui vaut 2 sur le disque d'aire π, donne aussi 2π.

**Ce que montrent les graphes :**

- Le disque D, son bord orienté dans le sens direct (flèches orange) et le champ V dessiné en quelques points (petites flèches grises).
- Le champ tourne autour de l'origine dans le même sens que le bord. Le produit V·dℓ est donc positif en tout point du cercle : la circulation est positive.

### Intégrale de sin(x)/x sur ]0, +∞[

**Saisie :** type « Intégrale », f = `sin(x)/x`, intervalle `]0, +oo[`.

<p align="center"><img src="docs/img/demo_leb_dirichlet.gif" width="760" alt="Déroulement de l'étude : Intégrale de sin(x)/x sur ]0, +∞["></p>

<p align="center"><img src="docs/img/leb_dirichlet_1.png" width="760" alt="Intégrale de sin(x)/x sur ]0, +∞[, étape 1"></p>

<p align="center"><img src="docs/img/leb_dirichlet_2.png" width="760" alt="Intégrale de sin(x)/x sur ]0, +∞[, étape 2"></p>

<p align="center"><img src="docs/img/leb_dirichlet_3.png" width="760" alt="Intégrale de sin(x)/x sur ]0, +∞[, étape 3"></p>

<p align="center"><img src="docs/img/leb_dirichlet_4.png" width="760" alt="Intégrale de sin(x)/x sur ]0, +∞[, étape 4"></p>

<p align="center"><img src="docs/img/leb_dirichlet_5.png" width="760" alt="Intégrale de sin(x)/x sur ]0, +∞[, étape 5"></p>

**Résultat :** L'intégrale généralisée converge (vers π/2), alors que l'intégrale de |f| diverge : elle converge sans converger absolument.

**Ce que montrent les graphes :**

- Courbe de f avec l'aire colorée entre la courbe et l'axe. Les arches au-dessus de l'axe comptent positivement, celles au-dessous négativement. La fenêtre est limitée à [0, 12] parce que l'intervalle est infini.
- G(X) = ∫ de 0 à X de f, tracée en fonction de X (échelle logarithmique). G oscille puis se stabilise près de 1,57, c'est-à-dire π/2.
- |f| en échelle log-log. Les points suivent à peu près une droite de pente proche de −1, comme 1/x. Une décroissance en 1/x est trop lente pour que l'intégrale de |f| converge.

### Intégrale curviligne le long d'une hélice, champ V = (yz, xz, xy)

**Saisie :** type « Intégrale curviligne / Green–Riemann », champ `y*z, x*z, x*y`, arc `x=cos(t), y=sin(t), z=t, t in [0,2*pi]`.

<p align="center"><img src="docs/img/curv_helice_1.png" width="760" alt="Intégrale curviligne le long d'une hélice, champ V = (yz, xz, xy), étape 1"></p>

<p align="center"><img src="docs/img/curv_helice_2.png" width="760" alt="Intégrale curviligne le long d'une hélice, champ V = (yz, xz, xy), étape 2"></p>

<p align="center"><img src="docs/img/curv_helice_3.png" width="760" alt="Intégrale curviligne le long d'une hélice, champ V = (yz, xz, xy), étape 3"></p>

<p align="center"><img src="docs/img/curv_helice_4.png" width="760" alt="Intégrale curviligne le long d'une hélice, champ V = (yz, xz, xy), étape 4"></p>

**Résultat :** V est le gradient de φ(x, y, z) = xyz. L'intégrale vaut φ(arrivée) − φ(départ) = 0 − 0 = 0. Le calcul numérique donne la même valeur.

**Ce que montrent les graphes :**

- La courbe dans l'espace : une hélice qui fait un tour en montant de z = 0 à z = 2π. Le point vert est le départ (1, 0, 0), le carré rouge l'arrivée (1, 0, 2π).

### Intégrale double de x² + y² sur le disque unité

**Saisie :** type « Intégrale multiple », f = `x**2+y**2`, domaine `x^2+y^2<=1`.

<p align="center"><img src="docs/img/multi_disque_1.png" width="760" alt="Intégrale double de x² + y² sur le disque unité, étape 1"></p>

<p align="center"><img src="docs/img/multi_disque_2.png" width="760" alt="Intégrale double de x² + y² sur le disque unité, étape 2"></p>

<p align="center"><img src="docs/img/multi_disque_3.png" width="760" alt="Intégrale double de x² + y² sur le disque unité, étape 3"></p>

<p align="center"><img src="docs/img/multi_disque_4.png" width="760" alt="Intégrale double de x² + y² sur le disque unité, étape 4"></p>

**Résultat :** On passe en coordonnées polaires : x = r cos θ, y = r sin θ, le jacobien vaut r et l'intégrande devient r³. L'intégrale vaut 2π × 1/4 = π/2.

**Ce que montrent les graphes :**

- À gauche, le domaine D découpé en tranches verticales (traits orangés). À droite, la surface z = f(x, y) au-dessus de D, en forme de cuvette. L'intégrale double est le volume sous cette surface.

### Suite uₙ = (1 + 1/n)ⁿ

**Saisie :** type « Suite », terme général `(1+1/n)**n`.

<p align="center"><img src="docs/img/an_suite_e_1.png" width="760" alt="Suite uₙ = (1 + 1/n)ⁿ, étape 1"></p>

<p align="center"><img src="docs/img/an_suite_e_2.png" width="760" alt="Suite uₙ = (1 + 1/n)ⁿ, étape 2"></p>

**Résultat :** La suite converge vers e.

**Ce que montrent les graphes :**

- Les points (n, uₙ). Ils montent vite puis se stabilisent près de e ≈ 2,718 : on lit la limite sur le graphe.

### Suite uₙ = n²/(n+1)

**Saisie :** type « Suite », terme général `n**2/(n+1)`.

<p align="center"><img src="docs/img/an_suite_div_1.png" width="760" alt="Suite uₙ = n²/(n+1), étape 1"></p>

<p align="center"><img src="docs/img/an_suite_div_2.png" width="760" alt="Suite uₙ = n²/(n+1), étape 2"></p>

**Résultat :** La limite est +∞ : la suite diverge.

**Ce que montrent les graphes :**

- Les points (n, uₙ) s'alignent sur une droite qui monte sans s'arrêter. Aucune valeur limite n'apparaît.

### Série de terme général 1/n²

**Saisie :** type « Série », terme général `1/n**2`, indice 1.

<p align="center"><img src="docs/img/an_serie_riemann_1.png" width="760" alt="Série de terme général 1/n², étape 1"></p>

<p align="center"><img src="docs/img/an_serie_riemann_2.png" width="760" alt="Série de terme général 1/n², étape 2"></p>

<p align="center"><img src="docs/img/an_serie_riemann_3.png" width="760" alt="Série de terme général 1/n², étape 3"></p>

**Résultat :** La série converge (critère de Riemann, car 2 > 1).

**Ce que montrent les graphes :**

- Les sommes partielles Sₙ = 1 + 1/4 + … + 1/n². Elles montent vite puis se stabilisent près de 1,64.
- Les termes 1/n², qui tendent vers 0 très rapidement.

### Série harmonique, terme général 1/n

**Saisie :** type « Série », terme général `1/n`, indice 1.

<p align="center"><img src="docs/img/an_serie_harmo_1.png" width="760" alt="Série harmonique, terme général 1/n, étape 1"></p>

<p align="center"><img src="docs/img/an_serie_harmo_2.png" width="760" alt="Série harmonique, terme général 1/n, étape 2"></p>

<p align="center"><img src="docs/img/an_serie_harmo_3.png" width="760" alt="Série harmonique, terme général 1/n, étape 3"></p>

**Résultat :** La série diverge (critère de Riemann, car 1 ≤ 1).

**Ce que montrent les graphes :**

- Les sommes partielles continuent de monter, de plus en plus lentement, sans se stabiliser.
- Les termes 1/n tendent vers 0. Cela ne suffit pas pour que la série converge.

### Série alternée de terme général (−1)ⁿ/n

**Saisie :** type « Série », terme général `(-1)**n/n`, indice 1.

<p align="center"><img src="docs/img/demo_an_serie_alt.gif" width="760" alt="Déroulement de l'étude : Série alternée de terme général (−1)ⁿ/n"></p>

<p align="center"><img src="docs/img/an_serie_alt_1.png" width="760" alt="Série alternée de terme général (−1)ⁿ/n, étape 1"></p>

<p align="center"><img src="docs/img/an_serie_alt_2.png" width="760" alt="Série alternée de terme général (−1)ⁿ/n, étape 2"></p>

<p align="center"><img src="docs/img/an_serie_alt_3.png" width="760" alt="Série alternée de terme général (−1)ⁿ/n, étape 3"></p>

<p align="center"><img src="docs/img/an_serie_alt_4.png" width="760" alt="Série alternée de terme général (−1)ⁿ/n, étape 4"></p>

**Résultat :** La série converge (critère de Leibniz : le signe alterne et |uₙ| = 1/n décroît vers 0).

**Ce que montrent les graphes :**

- Les sommes partielles oscillent de part et d'autre d'une valeur proche de −0,69 en se resserrant. Les sommes d'indice impair sont en dessous, celles d'indice pair au-dessus.
- Les termes (−1)ⁿ/n alternent de signe et leur valeur absolue tend vers 0.

### Série de terme général 1/(n ln²n)

**Saisie :** type « Série », terme général `1/(n*log(n)**2)`, indice 2.

<p align="center"><img src="docs/img/an_serie_bertrand_1.png" width="760" alt="Série de terme général 1/(n ln²n), étape 1"></p>

<p align="center"><img src="docs/img/an_serie_bertrand_2.png" width="760" alt="Série de terme général 1/(n ln²n), étape 2"></p>

<p align="center"><img src="docs/img/an_serie_bertrand_3.png" width="760" alt="Série de terme général 1/(n ln²n), étape 3"></p>

**Résultat :** La série converge (série de Bertrand, avec l'exposant 2 > 1 sur le logarithme).

**Ce que montrent les graphes :**

- Les sommes partielles croissent de plus en plus lentement.
- Les termes 1/(n ln²n) tombent vers 0.

### Suite de fonctions fₙ(x) = xⁿ sur [0, 1]

**Saisie :** type « Suite de fonctions », terme général `x**n`, intervalle `[0, 1]`.

<p align="center"><img src="docs/img/an_sf_xn_1.png" width="760" alt="Suite de fonctions fₙ(x) = xⁿ sur [0, 1], étape 1"></p>

<p align="center"><img src="docs/img/an_sf_xn_2.png" width="760" alt="Suite de fonctions fₙ(x) = xⁿ sur [0, 1], étape 2"></p>

**Résultat :** fₙ converge simplement vers 0 sur [0, 1[ et vers 1 en x = 1. La limite n'est pas continue alors que les fₙ le sont : la convergence n'est pas uniforme.

**Ce que montrent les graphes :**

- Les courbes de f₁ à f₂₀ et la limite (en noir). Chaque courbe reste proche de 0 sauf près de x = 1, où elle monte brusquement vers 1. La limite saute en 1.

### Série de fonctions de terme général xⁿ/n² sur [−1, 1]

**Saisie :** type « Série de fonctions », terme général `x**n/n**2`, indice 1, intervalle `[-1, 1]`.

<p align="center"><img src="docs/img/an_srf_xn2_1.png" width="760" alt="Série de fonctions de terme général xⁿ/n² sur [−1, 1], étape 1"></p>

<p align="center"><img src="docs/img/an_srf_xn2_2.png" width="760" alt="Série de fonctions de terme général xⁿ/n² sur [−1, 1], étape 2"></p>

<p align="center"><img src="docs/img/an_srf_xn2_3.png" width="760" alt="Série de fonctions de terme général xⁿ/n² sur [−1, 1], étape 3"></p>

<p align="center"><img src="docs/img/an_srf_xn2_4.png" width="760" alt="Série de fonctions de terme général xⁿ/n² sur [−1, 1], étape 4"></p>

<p align="center"><img src="docs/img/an_srf_xn2_5.png" width="760" alt="Série de fonctions de terme général xⁿ/n² sur [−1, 1], étape 5"></p>

**Résultat :** La série converge uniformément sur [−1, 1]. La somme est donc continue et on peut intégrer terme à terme.

**Ce que montrent les graphes :**

- Les sommes partielles S₁ à S₂₀ sur [−1, 1]. Les courbes se resserrent les unes contre les autres quand N augmente : l'écart devient petit en tous les points à la fois, ce qui illustre la convergence uniforme.

### Série entière ∑ xⁿ/n! sur [−2, 2]

**Saisie :** type « Série entière », terme général `x**n/factorial(n)`, indice 0, intervalle `[-2, 2]`.

<p align="center"><img src="docs/img/an_se_exp_1.png" width="760" alt="Série entière ∑ xⁿ/n! sur [−2, 2], étape 1"></p>

<p align="center"><img src="docs/img/an_se_exp_2.png" width="760" alt="Série entière ∑ xⁿ/n! sur [−2, 2], étape 2"></p>

<p align="center"><img src="docs/img/an_se_exp_3.png" width="760" alt="Série entière ∑ xⁿ/n! sur [−2, 2], étape 3"></p>

**Résultat :** La série converge absolument sur [−2, 2] et sa somme est eˣ.

**Ce que montrent les graphes :**

- Les sommes partielles S₀ à S₁₉ et la fonction eˣ (en noir). Plus N est grand, plus S_N colle à eˣ sur tout l'intervalle.

### Intégrale de x e^{−x} sur [0, +∞[

**Saisie :** type « Intégrale », f = `x*exp(-x)`, intervalle `[0, +oo[`.

<p align="center"><img src="docs/img/leb_x_exp_1.png" width="760" alt="Intégrale de x e^{−x} sur [0, +∞[, étape 1"></p>

<p align="center"><img src="docs/img/leb_x_exp_2.png" width="760" alt="Intégrale de x e^{−x} sur [0, +∞[, étape 2"></p>

<p align="center"><img src="docs/img/leb_x_exp_3.png" width="760" alt="Intégrale de x e^{−x} sur [0, +∞[, étape 3"></p>

<p align="center"><img src="docs/img/leb_x_exp_4.png" width="760" alt="Intégrale de x e^{−x} sur [0, +∞[, étape 4"></p>

<p align="center"><img src="docs/img/leb_x_exp_5.png" width="760" alt="Intégrale de x e^{−x} sur [0, +∞[, étape 5"></p>

**Résultat :** L'intégrale vaut 1 (intégration par parties, calculée jusqu'au bout).

**Ce que montrent les graphes :**

- Courbe de f avec l'aire colorée sous la courbe, sur la fenêtre [0, 12].
- G(X) = ∫ de 0 à X de f en fonction de X (échelle logarithmique). La courbe se stabilise sur la valeur 1, tracée en pointillés rouges.
- |f| en échelle log-log : la courbe plonge beaucoup plus vite que 1/x², donc f est intégrable en +∞.

### Intégrale de 1/x sur [1, +∞[

**Saisie :** type « Intégrale », f = `1/x`, intervalle `[1, +oo[`.

<p align="center"><img src="docs/img/demo_an_int_div.gif" width="760" alt="Déroulement de l'étude : Intégrale de 1/x sur [1, +∞["></p>

<p align="center"><img src="docs/img/an_int_div_1.png" width="760" alt="Intégrale de 1/x sur [1, +∞[, étape 1"></p>

<p align="center"><img src="docs/img/an_int_div_2.png" width="760" alt="Intégrale de 1/x sur [1, +∞[, étape 2"></p>

<p align="center"><img src="docs/img/an_int_div_3.png" width="760" alt="Intégrale de 1/x sur [1, +∞[, étape 3"></p>

<p align="center"><img src="docs/img/an_int_div_4.png" width="760" alt="Intégrale de 1/x sur [1, +∞[, étape 4"></p>

**Résultat :** ∫ de 1 à X de dx/x = ln X tend vers +∞ : l'intégrale diverge.

**Ce que montrent les graphes :**

- Courbe de f et aire colorée sur la fenêtre [1, 13].
- G(X) = ∫ de 1 à X de f en fonction de X (échelle logarithmique). La courbe monte sans se stabiliser.
- |f| en échelle log-log : une droite de pente −1. Une décroissance en 1/x est trop lente pour que l'aire reste finie.

### Intégrale double de xy sur 0 ≤ x ≤ 1, x ≤ y ≤ 2x

**Saisie :** type « Intégrale multiple », f = `x*y`, domaine `0<=x<=1, x<=y<=2*x`.

<p align="center"><img src="docs/img/multi_triangle_1.png" width="760" alt="Intégrale double de xy sur 0 ≤ x ≤ 1, x ≤ y ≤ 2x, étape 1"></p>

<p align="center"><img src="docs/img/multi_triangle_2.png" width="760" alt="Intégrale double de xy sur 0 ≤ x ≤ 1, x ≤ y ≤ 2x, étape 2"></p>

<p align="center"><img src="docs/img/multi_triangle_3.png" width="760" alt="Intégrale double de xy sur 0 ≤ x ≤ 1, x ≤ y ≤ 2x, étape 3"></p>

<p align="center"><img src="docs/img/multi_triangle_4.png" width="760" alt="Intégrale double de xy sur 0 ≤ x ≤ 1, x ≤ y ≤ 2x, étape 4"></p>

**Résultat :** On intègre d'abord en y de x à 2x, puis en x de 0 à 1 (Fubini). L'intégrale vaut 3/8. L'application refait le calcul dans l'autre ordre et retrouve 3/8.

**Ce que montrent les graphes :**

- À gauche, le domaine D : un triangle limité par les droites y = x, y = 2x et x = 1, découpé en tranches verticales. À droite, la surface z = xy au-dessus de D.

### Intégrale double de e^{−x²−y²} sur le plan

**Saisie :** type « Intégrale multiple », f = `exp(-x^2-y^2)`, domaine `R^2`.

<p align="center"><img src="docs/img/multi_cloche_1.png" width="760" alt="Intégrale double de e^{−x²−y²} sur le plan, étape 1"></p>

<p align="center"><img src="docs/img/multi_cloche_2.png" width="760" alt="Intégrale double de e^{−x²−y²} sur le plan, étape 2"></p>

<p align="center"><img src="docs/img/multi_cloche_3.png" width="760" alt="Intégrale double de e^{−x²−y²} sur le plan, étape 3"></p>

<p align="center"><img src="docs/img/multi_cloche_4.png" width="760" alt="Intégrale double de e^{−x²−y²} sur le plan, étape 4"></p>

<p align="center"><img src="docs/img/multi_cloche_5.png" width="760" alt="Intégrale double de e^{−x²−y²} sur le plan, étape 5"></p>

**Résultat :** Passage en coordonnées polaires. L'intégrale vaut π.

**Ce que montrent les graphes :**

- À gauche, le domaine : tout le plan, affiché sur [−45, 45]². À droite, la surface z = e^{−x²−y²} : un pic étroit au centre, presque nul partout ailleurs.

### Longueur de l'arc y = x² pour x de 0 à 1

**Saisie :** type « Intégrale curviligne / Green–Riemann », champ `longueur`, arc `x=t, y=t^2, t in [0,1]`.

<p align="center"><img src="docs/img/curv_longueur_parabole_1.png" width="760" alt="Longueur de l'arc y = x² pour x de 0 à 1, étape 1"></p>

<p align="center"><img src="docs/img/curv_longueur_parabole_2.png" width="760" alt="Longueur de l'arc y = x² pour x de 0 à 1, étape 2"></p>

<p align="center"><img src="docs/img/curv_longueur_parabole_3.png" width="760" alt="Longueur de l'arc y = x² pour x de 0 à 1, étape 3"></p>

**Résultat :** L = ∫ de 0 à 1 de √(1 + 4t²) dt = asinh(2)/4 + √5/2 ≈ 1,479.

**Ce que montrent les graphes :**

- L'arc γ. Le point vert est le départ (0, 0), le carré rouge l'arrivée (1, 1), la flèche orange indique le sens du parcours.

### Circulation de V = (2xy, x²) sur le cercle unité

**Saisie :** type « Intégrale curviligne / Green–Riemann », champ `2*x*y, x^2`, arc `x=cos(t), y=sin(t), t in [0,2*pi]`.

<p align="center"><img src="docs/img/curv_prevu_gradient_1.png" width="760" alt="Circulation de V = (2xy, x²) sur le cercle unité, étape 1"></p>

<p align="center"><img src="docs/img/curv_prevu_gradient_2.png" width="760" alt="Circulation de V = (2xy, x²) sur le cercle unité, étape 2"></p>

<p align="center"><img src="docs/img/curv_prevu_gradient_3.png" width="760" alt="Circulation de V = (2xy, x²) sur le cercle unité, étape 3"></p>

**Résultat :** Le résultat 0 est annoncé avant le calcul : V est le gradient de φ = x²y et la courbe est fermée. Le calcul numérique confirme 0.

**Ce que montrent les graphes :**

- Le cercle parcouru dans le sens direct, avec le champ V en gris. Le carré rouge en (1, 0) est le point d'arrivée, qui est aussi le point de départ.

### Intégrale de e^{−x²} sur ℝ

**Saisie :** type « Intégrale », f = `exp(-x**2)`, intervalle `]-oo, +oo[`.

<p align="center"><img src="docs/img/leb_cloche_1.png" width="760" alt="Intégrale de e^{−x²} sur ℝ, étape 1"></p>

<p align="center"><img src="docs/img/leb_cloche_2.png" width="760" alt="Intégrale de e^{−x²} sur ℝ, étape 2"></p>

<p align="center"><img src="docs/img/leb_cloche_3.png" width="760" alt="Intégrale de e^{−x²} sur ℝ, étape 3"></p>

<p align="center"><img src="docs/img/leb_cloche_4.png" width="760" alt="Intégrale de e^{−x²} sur ℝ, étape 4"></p>

<p align="center"><img src="docs/img/leb_cloche_5.png" width="760" alt="Intégrale de e^{−x²} sur ℝ, étape 5"></p>

**Résultat :** L'intégrale vaut √π, calculée en passant par l'intégrale double.

**Ce que montrent les graphes :**

- Courbe en cloche de f avec l'aire colorée, sur la fenêtre [−12, 12].
- G(X) = ∫ de 0 à X de f, tracée vers −∞ (à gauche) et vers +∞ (à droite). Chaque courbe se stabilise près de −0,89 et 0,89, soit −√π/2 et √π/2.
- |f| en échelle log-log, vers −∞ et vers +∞ : la chute est très rapide, bien plus qu'en 1/x².

### Suite fₙ(x) = n x e^{−n x²} sur [0, 1] : limite des intégrales et intégrale de la limite

**Saisie :** type « Suite de fonctions et intégrales », f_n = `n*x*exp(-n*x**2)`, intervalle `[0, 1]`.

<p align="center"><img src="docs/img/conv_nxexp_1.png" width="760" alt="Suite fₙ(x) = n x e^{−n x²} sur [0, 1] : limite des intégrales et intégrale de la limite, étape 1"></p>

<p align="center"><img src="docs/img/conv_nxexp_2.png" width="760" alt="Suite fₙ(x) = n x e^{−n x²} sur [0, 1] : limite des intégrales et intégrale de la limite, étape 2"></p>

<p align="center"><img src="docs/img/conv_nxexp_3.png" width="760" alt="Suite fₙ(x) = n x e^{−n x²} sur [0, 1] : limite des intégrales et intégrale de la limite, étape 3"></p>

<p align="center"><img src="docs/img/conv_nxexp_4.png" width="760" alt="Suite fₙ(x) = n x e^{−n x²} sur [0, 1] : limite des intégrales et intégrale de la limite, étape 4"></p>

<p align="center"><img src="docs/img/conv_nxexp_5.png" width="760" alt="Suite fₙ(x) = n x e^{−n x²} sur [0, 1] : limite des intégrales et intégrale de la limite, étape 5"></p>

**Résultat :** fₙ converge simplement vers 0, mais ∫ fₙ = (1 − e⁻ⁿ)/2 tend vers 1/2. La limite des intégrales (1/2) est différente de l'intégrale de la limite (0).

**Ce que montrent les graphes :**

- Les courbes f₁, f₂, f₃, f₅, f₁₀ et leur limite simple (pointillés rouges, la fonction nulle). Les bosses sont de plus en plus hautes et de plus en plus proches de 0. En chaque point, la valeur finit par retomber à 0.
- Les points ∫ fₙ pour n = 1 à 10 se rapprochent de 0,5. La droite rouge à 0 est l'intégrale de la limite. L'écart entre les deux montre qu'on ne peut pas échanger limite et intégrale ici.

### Aire de l'ellipse x²/4 + y²/9 ≤ 1 par Green–Riemann

**Saisie :** type « Intégrale curviligne / Green–Riemann », champ `aire`, domaine `x^2/4+y^2/9<=1`.

<p align="center"><img src="docs/img/curv_aire_ellipse_1.png" width="760" alt="Aire de l'ellipse x²/4 + y²/9 ≤ 1 par Green–Riemann, étape 1"></p>

<p align="center"><img src="docs/img/curv_aire_ellipse_2.png" width="760" alt="Aire de l'ellipse x²/4 + y²/9 ≤ 1 par Green–Riemann, étape 2"></p>

<p align="center"><img src="docs/img/curv_aire_ellipse_3.png" width="760" alt="Aire de l'ellipse x²/4 + y²/9 ≤ 1 par Green–Riemann, étape 3"></p>

<p align="center"><img src="docs/img/curv_aire_ellipse_4.png" width="760" alt="Aire de l'ellipse x²/4 + y²/9 ≤ 1 par Green–Riemann, étape 4"></p>

<p align="center"><img src="docs/img/curv_aire_ellipse_5.png" width="760" alt="Aire de l'ellipse x²/4 + y²/9 ≤ 1 par Green–Riemann, étape 5"></p>

<p align="center"><img src="docs/img/curv_aire_ellipse_6.png" width="760" alt="Aire de l'ellipse x²/4 + y²/9 ≤ 1 par Green–Riemann, étape 6"></p>

**Résultat :** L'aire s'écrit ½ ∮ (x dy − y dx). Avec le paramétrage x = 2 cos t, y = 3 sin t, on trouve 6π. L'intégrale double de 1 sur D donne la même valeur.

**Ce que montrent les graphes :**

- L'ellipse D et son bord orienté dans le sens direct (flèches orange).

### Green–Riemann sur le triangle x ≥ 0, y ≥ 0, x + y ≤ 1, champ V = (y², x)

**Saisie :** type « Intégrale curviligne / Green–Riemann », champ `y^2, x`, domaine `x>=0, y>=0, x+y<=1`.

<p align="center"><img src="docs/img/curv_green_triangle_1.png" width="760" alt="Green–Riemann sur le triangle x ≥ 0, y ≥ 0, x + y ≤ 1, champ V = (y², x), étape 1"></p>

<p align="center"><img src="docs/img/curv_green_triangle_2.png" width="760" alt="Green–Riemann sur le triangle x ≥ 0, y ≥ 0, x + y ≤ 1, champ V = (y², x), étape 2"></p>

<p align="center"><img src="docs/img/curv_green_triangle_3.png" width="760" alt="Green–Riemann sur le triangle x ≥ 0, y ≥ 0, x + y ≤ 1, champ V = (y², x), étape 3"></p>

<p align="center"><img src="docs/img/curv_green_triangle_4.png" width="760" alt="Green–Riemann sur le triangle x ≥ 0, y ≥ 0, x + y ≤ 1, champ V = (y², x), étape 4"></p>

<p align="center"><img src="docs/img/curv_green_triangle_5.png" width="760" alt="Green–Riemann sur le triangle x ≥ 0, y ≥ 0, x + y ≤ 1, champ V = (y², x), étape 5"></p>

<p align="center"><img src="docs/img/curv_green_triangle_6.png" width="760" alt="Green–Riemann sur le triangle x ≥ 0, y ≥ 0, x + y ≤ 1, champ V = (y², x), étape 6"></p>

**Résultat :** Le bord est formé de trois segments, paramétrés un par un. La circulation est la somme des trois intégrales (relation de Chasles) et vaut 1/6. L'intégrale double de ∂Q/∂x − ∂P/∂y = 1 − 2y donne aussi 1/6.

**Ce que montrent les graphes :**

- Le triangle D, le champ V en gris et une flèche orange sur chaque côté, qui indique le sens de parcours (le domaine reste à gauche).

### Série entière ∑ (−1)ⁿ⁺¹ xⁿ/n sur [−2, 2]

**Saisie :** type « Série entière », terme général `(-1)**(n+1)*x**n/n`, indice 1, intervalle `[-2, 2]`.

<p align="center"><img src="docs/img/demo_an_se_ln.gif" width="760" alt="Déroulement de l'étude : Série entière ∑ (−1)ⁿ⁺¹ xⁿ/n sur [−2, 2]"></p>

<p align="center"><img src="docs/img/an_se_ln_1.png" width="760" alt="Série entière ∑ (−1)ⁿ⁺¹ xⁿ/n sur [−2, 2], étape 1"></p>

<p align="center"><img src="docs/img/an_se_ln_2.png" width="760" alt="Série entière ∑ (−1)ⁿ⁺¹ xⁿ/n sur [−2, 2], étape 2"></p>

<p align="center"><img src="docs/img/an_se_ln_3.png" width="760" alt="Série entière ∑ (−1)ⁿ⁺¹ xⁿ/n sur [−2, 2], étape 3"></p>

<p align="center"><img src="docs/img/an_se_ln_4.png" width="760" alt="Série entière ∑ (−1)ⁿ⁺¹ xⁿ/n sur [−2, 2], étape 4"></p>

<p align="center"><img src="docs/img/an_se_ln_5.png" width="760" alt="Série entière ∑ (−1)ⁿ⁺¹ xⁿ/n sur [−2, 2], étape 5"></p>

**Résultat :** Le rayon de convergence est 1. La série converge absolument sur ]−1, 1[ et diverge pour |x| > 1. Aux bords, elle diverge en −1 (série harmonique) et converge en 1 (série alternée). La somme est ln(1 + x).

**Ce que montrent les graphes :**

- Les sommes partielles S₁ à S₂₀ et la fonction ln(1 + x) en noir. Sur ]−1, 1[, elles s'en approchent. Au-delà de 1 et de −1, elles partent vers +∞ ou −∞ de plus en plus vite : la série diverge. La courbe noire s'arrête en x = −1, où ln(1 + x) n'est pas défini.

### Champ V = (−y, x)/(x² + y²) sur le cercle unité

**Saisie :** type « Intégrale curviligne / Green–Riemann », champ `-y/(x^2+y^2), x/(x^2+y^2)`, arc `x=cos(t), y=sin(t), t in [0,2*pi]`.

<p align="center"><img src="docs/img/curv_singulier_1.png" width="760" alt="Champ V = (−y, x)/(x² + y²) sur le cercle unité, étape 1"></p>

<p align="center"><img src="docs/img/curv_singulier_2.png" width="760" alt="Champ V = (−y, x)/(x² + y²) sur le cercle unité, étape 2"></p>

<p align="center"><img src="docs/img/curv_singulier_3.png" width="760" alt="Champ V = (−y, x)/(x² + y²) sur le cercle unité, étape 3"></p>

<p align="center"><img src="docs/img/curv_singulier_4.png" width="760" alt="Champ V = (−y, x)/(x² + y²) sur le cercle unité, étape 4"></p>

**Résultat :** ∂Q/∂x − ∂P/∂y = 0 en tout point sauf l'origine, et pourtant la circulation sur le cercle vaut 2π. V n'a donc pas de potentiel sur le plan privé de l'origine. L'application ne calcule pas par Green–Riemann, car V n'est pas défini en 0, point intérieur au disque.

**Ce que montrent les graphes :**

- Le cercle unité et le champ V : les flèches tournent autour de l'origine, point où le champ n'est pas défini.

### Série de Fourier de f(x) = |x| sur [−π, π]

**Saisie :** type « Série de Fourier », f = `abs(x)`, N = 6, intervalle `[-pi, pi]`.

<p align="center"><img src="docs/img/demo_fourier_abs.gif" width="760" alt="Déroulement de l'étude : Série de Fourier de f(x) = |x| sur [−π, π]"></p>

<p align="center"><img src="docs/img/fourier_abs_1.png" width="760" alt="Série de Fourier de f(x) = |x| sur [−π, π], étape 1"></p>

<p align="center"><img src="docs/img/fourier_abs_2.png" width="760" alt="Série de Fourier de f(x) = |x| sur [−π, π], étape 2"></p>

<p align="center"><img src="docs/img/fourier_abs_3.png" width="760" alt="Série de Fourier de f(x) = |x| sur [−π, π], étape 3"></p>

<p align="center"><img src="docs/img/fourier_abs_4.png" width="760" alt="Série de Fourier de f(x) = |x| sur [−π, π], étape 4"></p>

<p align="center"><img src="docs/img/fourier_abs_5.png" width="760" alt="Série de Fourier de f(x) = |x| sur [−π, π], étape 5"></p>

<p align="center"><img src="docs/img/fourier_abs_6.png" width="760" alt="Série de Fourier de f(x) = |x| sur [−π, π], étape 6"></p>

<p align="center"><img src="docs/img/fourier_abs_7.png" width="760" alt="Série de Fourier de f(x) = |x| sur [−π, π], étape 7"></p>

<p align="center"><img src="docs/img/fourier_abs_8.png" width="760" alt="Série de Fourier de f(x) = |x| sur [−π, π], étape 8"></p>

<p align="center"><img src="docs/img/fourier_abs_9.png" width="760" alt="Série de Fourier de f(x) = |x| sur [−π, π], étape 9"></p>

<p align="center"><img src="docs/img/fourier_abs_10.png" width="760" alt="Série de Fourier de f(x) = |x| sur [−π, π], étape 10"></p>

<p align="center"><img src="docs/img/fourier_abs_11.png" width="760" alt="Série de Fourier de f(x) = |x| sur [−π, π], étape 11"></p>

**Résultat :** f est paire, donc tous les bₙ sont nuls. Les aₙ sont nuls pour n pair et valent −4/(πn²) pour n impair. La série converge vers f en tout point.

**Ce que montrent les graphes :**

- La fonction |x| prolongée en fonction 2π-périodique (en noir) et les sommes partielles S₁, S₃, S₆ et S₁₂. Plus on ajoute de termes, plus la somme colle à la fonction, surtout près des coins.
- À gauche, les coefficients aₙ (barres), puis les bₙ (tous nuls), puis l'amplitude des coefficients en échelle log-log : les points suivent une droite de pente −2, c'est-à-dire que les coefficients décroissent comme 1/n².
- À gauche, l'écart entre f et Sₙ en fonction de N : écart moyen (violet) et écart maximal (rouge), qui tendent tous deux vers 0. À droite, la part de l'énergie de f déjà contenue dans Sₙ : elle est proche de 1 dès les premiers termes.
- Une carte de l'erreur |f(x) − Sₙ(x)| : x en abscisse, N en ordonnée, la couleur donne le logarithme de l'erreur (plus sombre = plus petite). L'erreur reste la plus grande en x = 0 et x = ±π, aux points anguleux.
- L'erreur en chaque point d'une période, pour S₁, S₃, S₆ et S₂₄, en échelle logarithmique. Les creux sont les points où l'erreur s'annule.

## Types d'étude

| Type | Ce qu'on saisit | Ce que l'application fait |
|---|---|---|
| Suite | terme général uₙ | limite, conclusion, graphe |
| Série | uₙ et indice de départ | critères de convergence, justification, graphes des termes et des sommes partielles |
| Suite de fonctions | fₙ(x) et intervalle | limite, convergence simple et uniforme, graphes |
| Série de fonctions | uₙ(x), indice, intervalle | convergence, sommes partielles tracées |
| Série entière | aₙxⁿ, indice, intervalle | rayon de convergence, étude sur l'intervalle et aux bords, fonction somme |
| Développement en série entière | f(x), ordre | série, rayon, polynômes de Taylor tracés |
| Série de Fourier | f(x) périodique, intervalle | coefficients, convergence, sommes partielles, erreurs |
| Intégrale | f(x), intervalle | points à problème, convergence, calcul en égalités alignées |
| Suite de fonctions et intégrales | fₙ(x), intervalle | limite des intégrales et intégrale de la limite |
| Intégrale multiple | f, domaine par inégalités | Fubini, changement de variables, graphes |
| Intégrale curviligne et Green–Riemann | champ (P, Q) ou (P, Q, R), arc ou domaine | circulation, longueur, aire, vérification de Green–Riemann |

Critères utilisés pour les séries numériques : divergence grossière, Riemann, Bertrand, d'Alembert, Cauchy (racine), Leibniz, comparaison, équivalents, convergence absolue.

## Installation

Python 3.10 ou plus récent.

```bash
git clone https://github.com/Prospere-dev/convergence-assistant.git
cd convergence-assistant
python -m venv .venv
# Windows : .venv\Scripts\activate      Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Ouvrir ensuite <http://127.0.0.1:5000>. MathJax est chargé depuis un CDN, il faut donc une connexion internet pour afficher les formules.

Notation : `^` ou `**` pour les puissances, `oo` pour +∞ et `-oo` pour −∞ dans les bornes. Un intervalle ouvert peut s'écrire `] a , b [`.

## Fichiers

```
app.py                cœur du calcul (SymPy) et route Flask /compute
lebesgue.py           intégrales d'une variable
multi.py              intégrales multiples
fourier.py            séries de Fourier
curv.py               intégrales curvilignes et Green–Riemann
templates/index.html  formulaire et zone de résultats
static/app.js         envoi de la saisie, affichage des messages
static/style.css      mise en forme
test_app.py           tests pytest
```

Le navigateur envoie la saisie en JSON à `/compute`. Le serveur renvoie une liste de messages (formule, information, erreur, graphe). Le JavaScript les affiche, MathJax met en forme le LaTeX, et les graphes arrivent en PNG.

## Tests

```bash
pip install pytest
python -m pytest -v
```

## Limites

- Le calcul est symbolique (SymPy). Sur certaines expressions, SymPy ne conclut pas. L'application doit alors le dire sans donner de verdict.
- Les graphes montrent un nombre fini de termes. Ils illustrent un résultat, ils ne le démontrent pas.
- Usage local seulement. Les expressions saisies passent par `sympify`, qui peut exécuter du code Python. Ne pas mettre l'application sur un serveur public sans restreindre l'analyse des entrées et sans retirer `debug=True`.

## Licence

MIT. Voir le fichier [LICENSE](LICENSE).
