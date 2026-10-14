# =============================================================
# curv.py : intégrales curvilignes et théorème de Green–Riemann
#   - circulation d'un champ  ∫_γ P dx + Q dy (+ R dz)
#   - intégrale d'une fonction  ∫_γ f ds, longueur, aire
#   - champs conservatifs (potentiel), Green–Riemann sur un domaine D
# =============================================================
import re
import numpy as np
import sympy as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lebesgue as _L

st = None  # renseigné par app.py : curv.st = st

X, Y, Z = sp.symbols("x y z", real=True)


class _Abandon(Exception):
    pass


def _push():
    _L.st = st


def _P(s):
    _push()
    _L._P(s)


def _T(s):
    return _L._T(s)


def _titre(s):
    _push()
    _L._titre(s)


def _rouge(s):
    _push()
    _L._rouge(s)


def _lt(s):
    """Texte (avec $maths$) affiché en une ligne."""
    _P(_T(s))


# -------------------------------------------------------------
# Analyse syntaxique
# -------------------------------------------------------------
def _decouper(s, seps=",;"):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch in seps and depth <= 0:
            parts.append(cur)
            cur = ""
            depth = 0
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def _nettoie(s):
    s = (s or "").strip()
    s = s.replace("∈", " in ").replace("≤", "<=").replace("≥", ">=").replace("−", "-").replace("π", "pi")
    s = s.replace("²", "**2").replace("³", "**3").replace("·", "*")
    return s


def _enleve_parentheses(s):
    s = s.strip()
    while len(s) >= 2 and s[0] in "([" and s[-1] in ")]":
        depth = 0
        ok = True
        for i, ch in enumerate(s):
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            if depth == 0 and i < len(s) - 1:
                ok = False
                break
        if not ok:
            break
        s = s[1:-1].strip()
    return s


def _sym(expr_str, loc, extra=None):
    d = dict(loc)
    d.update({"x": X, "y": Y, "z": Z})
    if extra:
        d.update(extra)
    return sp.sympify(expr_str, locals=d)


def _parse_champ(s, loc):
    """Renvoie ('aire'|'longueur'|'scalaire'|'champ', composantes)."""
    s0 = _nettoie(s)
    bas = s0.lower().strip()
    if bas in ("aire", "area"):
        return "aire", []
    if bas in ("longueur", "length", "l"):
        return "longueur", []
    s1 = _enleve_parentheses(s0)
    parts = _decouper(s1)
    if len(parts) == 1:
        return "scalaire", [_sym(parts[0], loc)]
    if len(parts) in (2, 3):
        return "champ", [_sym(p, loc) for p in parts]
    raise _Abandon("Le champ doit avoir 2 ou 3 composantes, séparées par des virgules.")


_RE_PARAM_IN = re.compile(r"(?:^|[,;\s])([A-Za-z_]\w*)\s+(?:in|dans)\s*([\[\(\]])\s*(.+?)\s*[;,]\s*(.+?)\s*([\]\)\[])\s*$")
_RE_PARAM_CH = re.compile(r"(?:^|[,;])\s*([^,;<>=]+?)\s*<=?\s*([A-Za-z_]\w*)\s*<=?\s*([^,;<>=]+?)\s*$")


def _parse_courbe(s, loc):
    """Courbe paramétrée : renvoie (t, a, b, [x(t), y(t)(, z(t))], noms) ou None si ce n'est pas une courbe."""
    s0 = _nettoie(s)
    m = _RE_PARAM_IN.search(s0)
    tnom = a = b = None
    reste = s0
    if m:
        tnom, a, b = m.group(1), m.group(3), m.group(4)
        reste = s0[:m.start()] if m.start() > 0 else ""
        reste = reste.rstrip(" ,;")
    else:
        m = _RE_PARAM_CH.search(s0)
        avant0 = s0[:m.start()] if m else ""
        if m and re.search(r"(?<![<>=])=(?!=)", avant0) and re.fullmatch(r"[A-Za-z_]\w*", m.group(2)):
            # a <= t <= b, éventuellement précédé des composantes
            avant = s0[:m.start()].rstrip(" ,;")
            tnom, a, b = m.group(2), m.group(1), m.group(3)
            reste = avant
        else:
            m2 = re.search(r"(?:^|[,;\s])([A-Za-z_]\w*)\s*(?:de|from)\s+(.+?)\s+(?:à|a|to)\s+(.+?)\s*$", s0) if re.search(r"(?<![<>=])=(?!=)", s0) else None
            if m2:
                tnom, a, b = m2.group(1), m2.group(2), m2.group(3)
                reste = s0[:m2.start()].rstrip(" ,;")
    if tnom is None:
        return None
    if tnom in ("x", "y", "z") and not re.search(r"(?<![<>=])=(?!=)", reste):
        return None   # « x in [0,2], y in [0,x] » : c'est un domaine, pas une courbe
    t = sp.Symbol(tnom, real=True)
    extra = {tnom: t}
    # le paramètre ne doit pas être écrasé par les symboles x, y, z
    d = dict(loc)
    d.update({"x": X, "y": Y, "z": Z})
    d[tnom] = t
    comps = {}
    ordre = []
    for p in _decouper(_enleve_parentheses(reste)):
        mm = re.match(r"^\s*([xyz])\s*(?:\(\s*\w+\s*\))?\s*=\s*(.+)$", p)
        if mm:
            comps[mm.group(1)] = sp.sympify(mm.group(2), locals=d)
        else:
            ordre.append(sp.sympify(p, locals=d))
    if ordre and comps:
        raise _Abandon("Mélange de composantes nommées et positionnelles : choisissez l'une des deux écritures.")
    if ordre:
        if len(ordre) not in (2, 3):
            raise _Abandon("La courbe doit avoir 2 ou 3 composantes.")
        noms = "xyz"[:len(ordre)]
        comps = {noms[i]: ordre[i] for i in range(len(ordre))}
    if tnom in ("x", "y") and tnom not in comps:
        # graphe explicite  y=f(x), x in [a,b]
        autre = "y" if tnom == "x" else "x"
        if autre in comps:
            comps[tnom] = t
    if "x" not in comps or "y" not in comps:
        raise _Abandon("La courbe doit donner x et y (et éventuellement z).")
    liste = [comps["x"], comps["y"]] + ([comps["z"]] if "z" in comps else [])
    # les noms x, y, z de la courbe sont des fonctions de t : on les remplace par les symboles du paramètre
    a_ = sp.sympify(a, locals=d)
    b_ = sp.sympify(b, locals=d)
    return t, a_, b_, liste, tnom


def _parse_domaine(s, loc):
    """Contraintes sur (x, y) : liste d'expressions g telles que g >= 0."""
    s0 = _nettoie(s)
    s0 = re.sub(r"\bet\b|\band\b", ",", s0)
    cons = []
    global _TXT_DOM
    _TXT_DOM = []
    for piece in _decouper(s0):
        m = re.fullmatch(r"\s*([xy])\s+in\s*[\[\(\]]\s*(.+?)\s*[;,]\s*(.+?)\s*[\]\)\[]\s*", piece)
        if m:
            v = X if m.group(1) == "x" else Y
            cons.append(v - _sym(m.group(2), loc))
            cons.append(_sym(m.group(3), loc) - v)
            _TXT_DOM.append(sp.latex(_sym(m.group(2), loc)) + r" \leq " + sp.latex(v) + r" \leq " + sp.latex(_sym(m.group(3), loc)))
            continue
        ops = re.split(r"(<=|>=|<|>)", piece)
        if len(ops) < 3:
            raise _Abandon("« %s » n'est ni une inégalité (domaine D) ni une courbe : pour un arc, ajoutez l'intervalle du paramètre, par exemple t in [0,1]." % piece)
        exprs = [_sym(ops[i], loc) for i in range(0, len(ops), 2)]
        syms = ops[1::2]
        _lt_ = {"<=": r" \leq ", ">=": r" \geq ", "<": " < ", ">": " > "}
        _TXT_DOM.append(sp.latex(exprs[0]) + "".join(_lt_[o] + sp.latex(exprs[i + 1]) for i, o in enumerate(syms)))
        for i, o in enumerate(syms):
            gauche, droite = exprs[i], exprs[i + 1]
            cons.append((droite - gauche) if o in ("<=", "<") else (gauche - droite))
    return [sp.simplify(c) for c in cons]


# -------------------------------------------------------------
# Outils de calcul
# -------------------------------------------------------------
def _speciale(e):
    try:
        return bool(e.has(*_L._SPECF))
    except Exception:
        return False


def _val(g, t, a, b):
    """Valeur exacte de ∫_a^b g dt (None si introuvable ou non élémentaire)."""
    def calc():
        r = sp.integrate(g, (t, a, b))
        return sp.simplify(r)
    r = _L._avec_delai(calc, 25)
    if r is None:
        return None
    try:
        if r.has(sp.Integral) or r.has(sp.nan) or r.has(sp.zoo) or r.has(sp.oo) or r.has(-sp.oo) or _speciale(r):
            return None
        if r.has(sp.I) and not g.has(sp.I):
            r = sp.simplify(sp.re(r))
    except Exception:
        return None
    return r


def _borne_lat(b):
    return _L._borne_lat(b)


def _ligne_integrale(g, t, a, b, valeur=None):
    """Affiche la chaîne d'égalités alignées de ∫_a^b g dt et renvoie la valeur."""
    v = valeur if valeur is not None else _val(g, t, a, b)
    if v is None:
        raise _Abandon("Le calcul exact de l'intégrale %s n'aboutit pas." % sp.sstr(g))
    bloc = None
    try:
        bloc = _L._chaine_int(g, t, a, b, False, False, v)
    except Exception:
        bloc = None
    if bloc:
        _P(bloc)
    else:
        _P(r"\int_{" + _borne_lat(a) + "}^{" + _borne_lat(b) + "}" + sp.latex(g) + r"\,d" + sp.latex(t) + " = " + sp.latex(v))
    return v


def _simp(e):
    try:
        r = sp.simplify(e)
        return r if len(str(r)) <= len(str(e)) + 8 else e
    except Exception:
        return e


def _norme_vitesse(dc, t):
    """‖γ'‖ simplifiée (le radicande est un carré parfait dans les cas classiques)."""
    r2 = sp.simplify(sum(c ** 2 for c in dc))
    try:
        r2 = sp.trigsimp(r2)
    except Exception:
        pass
    n = sp.sqrt(r2)
    try:
        n = sp.simplify(sp.powsimp(sp.powdenest(n, force=True), force=True))
    except Exception:
        pass
    return r2, n


def _fini_sur(exprs, vars_, boites, n=600, seed=7):
    """Vrai si toutes les expressions sont finies et réelles en n points de la boîte (hypothèse de régularité)."""
    rng = np.random.default_rng(seed)
    try:
        fs = [sp.lambdify(vars_, e, "numpy") for e in exprs]
    except Exception:
        return False
    pts = [rng.uniform(lo, hi, n) for (lo, hi) in boites]
    # grille régulière contenant les points « remarquables » (0, entiers, demi-entiers)
    try:
        axes = [np.linspace(lo, hi, 13 if len(boites) <= 2 else 7) for (lo, hi) in boites]
        axes = [np.unique(np.concatenate([a_, np.array([0.0]) if a_.min() <= 0 <= a_.max() else np.array([])])) for a_ in axes]
        G = np.meshgrid(*axes, indexing="ij")
        pts = [np.concatenate([p, g.ravel()]) for p, g in zip(pts, G)]
        n = len(pts[0])
    except Exception:
        pass
    for f in fs:
        with np.errstate(all="ignore"):
            try:
                v = np.asarray(f(*pts), dtype=complex)
            except Exception:
                return False
        if v.ndim == 0:
            v = np.full(n, v)
        if not np.all(np.isfinite(v)) or np.max(np.abs(v.imag)) > 1e-9:
            return False
    return True


def _lisse_global(exprs, vars_):
    """Régularité C¹ sur tout l'espace (aucune singularité dans une grande boîte) et aucun |.|, signe, morceaux."""
    for e in exprs:
        if e.has(sp.Abs, sp.sign, sp.Piecewise, sp.floor, sp.ceiling, sp.Max, sp.Min):
            return False
    return _fini_sur(exprs, vars_, [(-6, 6)] * len(vars_), 1500) and _fini_sur(exprs, vars_, [(-3, 3)] * len(vars_), 800)


def _num_int(g, t, a, b):
    try:
        from scipy.integrate import quad
        f = sp.lambdify(t, g, "numpy")
        a_, b_ = float(sp.N(a)), float(sp.N(b))
        v, err = quad(lambda s: float(np.real(f(s))), a_, b_, limit=200)
        return v
    except Exception:
        return None


def _fmt_num(v):
    if v is None or not np.isfinite(v):
        return None
    return "0" if abs(v) < 1e-9 else ("%.10g" % v)


# -------------------------------------------------------------
# Graphes
# -------------------------------------------------------------
def _graphe_courbe(t, a, b, comp, champ=None):
    try:
        fa, fb = float(sp.N(a)), float(sp.N(b))
        ts = np.linspace(fa, fb, 600)
        fx = [sp.lambdify(t, c, "numpy") for c in comp]
        pts = [np.asarray(f(ts), dtype=float) * np.ones_like(ts) for f in fx]
    except Exception:
        return
    if len(comp) == 2:
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        xs, ys = pts
        if champ is not None:
            try:
                dx = xs.max() - xs.min() or 1.0
                dy = ys.max() - ys.min() or 1.0
                gx = np.linspace(xs.min() - 0.15 * dx, xs.max() + 0.15 * dx, 13)
                gy = np.linspace(ys.min() - 0.15 * dy, ys.max() + 0.15 * dy, 13)
                GX, GY = np.meshgrid(gx, gy)
                Fp = sp.lambdify((X, Y), champ[0], "numpy")
                Fq = sp.lambdify((X, Y), champ[1], "numpy")
                with np.errstate(all="ignore"):
                    U = np.asarray(Fp(GX, GY), dtype=float) * np.ones_like(GX)
                    V = np.asarray(Fq(GX, GY), dtype=float) * np.ones_like(GX)
                N = np.hypot(U, V)
                N[N == 0] = 1
                ok = np.isfinite(N)
                ax.quiver(GX[ok], GY[ok], (U / N)[ok], (V / N)[ok], color="#bdbdbd", angles="xy", width=0.003)
            except Exception:
                pass
        ax.plot(xs, ys, color="#08519c", lw=2.2)
        k = len(ts) // 2
        j = max(k - 4, 0)
        ax.annotate("", xy=(xs[k], ys[k]), xytext=(xs[j], ys[j]), arrowprops=dict(arrowstyle="-|>", color="#d94801", lw=2, mutation_scale=20))
        ax.plot([xs[0]], [ys[0]], "o", color="#238b45", ms=7, label=r"départ $\gamma(a)$")
        ax.plot([xs[-1]], [ys[-1]], "s", color="#cb181d", ms=7, label=r"arrivée $\gamma(b)$")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(r"Courbe $\gamma$ et sens de parcours")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        fig = plt.figure(figsize=(6.4, 5.4))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(*pts, color="#08519c", lw=2.2)
        ax.plot([pts[0][0]], [pts[1][0]], [pts[2][0]], "o", color="#238b45", ms=7)
        ax.plot([pts[0][-1]], [pts[1][-1]], [pts[2][-1]], "s", color="#cb181d", ms=7)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")
        ax.set_title(r"Courbe $\gamma$ (départ vert, arrivée rouge)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def _graphe_domaine(pieces, champ=None):
    """pieces : liste de (xs, ys) en ordre de parcours positif."""
    try:
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        allx = np.concatenate([p[0] for p in pieces])
        ally = np.concatenate([p[1] for p in pieces])
        ax.fill(allx, ally, color="#9ecae1", alpha=0.55)
        for xs, ys in pieces:
            ax.plot(xs, ys, color="#08519c", lw=2.2)
            k = len(xs) // 2
            j = max(k - 3, 0)
            if k > j and (xs[k] != xs[j] or ys[k] != ys[j]):
                ax.annotate("", xy=(xs[k], ys[k]), xytext=(xs[j], ys[j]), arrowprops=dict(arrowstyle="-|>", color="#d94801", lw=2, mutation_scale=20))
        if champ is not None:
            dx = allx.max() - allx.min() or 1.0
            dy = ally.max() - ally.min() or 1.0
            gx = np.linspace(allx.min(), allx.max(), 11)
            gy = np.linspace(ally.min(), ally.max(), 11)
            GX, GY = np.meshgrid(gx, gy)
            try:
                Fp = sp.lambdify((X, Y), champ[0], "numpy")
                Fq = sp.lambdify((X, Y), champ[1], "numpy")
                with np.errstate(all="ignore"):
                    U = np.asarray(Fp(GX, GY), dtype=float) * np.ones_like(GX)
                    V = np.asarray(Fq(GX, GY), dtype=float) * np.ones_like(GX)
                N = np.hypot(U, V)
                N[N == 0] = 1
                ok = np.isfinite(N)
                ax.quiver(GX[ok], GY[ok], (U / N)[ok], (V / N)[ok], color="#6a6a6a", angles="xy", width=0.003)
            except Exception:
                pass
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(r"Domaine $D$, bord $\partial D$ orienté positivement")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        plt.close("all")


# -------------------------------------------------------------
# Intégrale curviligne le long d'un arc paramétré
# -------------------------------------------------------------
def _par(e):
    return r"\left(" + sp.latex(e) + r"\right)" if (e.is_Add or e.could_extract_minus_sign()) else sp.latex(e)


def _par_add(e):
    return r"\left(" + sp.latex(e) + r"\right)" if e.is_Add else sp.latex(e)


def _notations(dim, kind):
    _titre("Notations")
    comp = "(P,Q)" if dim == 2 else "(P,Q,R)"
    if kind == "champ":
        _lt("$\\vec V=%s$ est le champ de vecteurs ; $P$, $Q$%s sont ses composantes." % (comp, ", $R$" if dim == 3 else ""))
    if kind == "domaine":
        _lt("$\\vec V=(P,Q)$ est le champ de vecteurs ; $D$ est le domaine et $\\partial D^+$ son bord orienté positivement (en parcourant le bord, $D$ reste à gauche).")
        _lt("Un morceau du bord est paramétré par $\\gamma(t)=(x(t),y(t))$, $t\\in[a,b]$ ; $\\gamma'(t)$ est son vecteur vitesse.")
    else:
        _lt("L'arc est paramétré par $\\gamma(t)$, $t\\in[a,b]$ ; $\\gamma'(t)$ est son vecteur vitesse.")
    if kind in ("champ", "domaine"):
        _lt("$d\\vec\\ell=\\gamma'(t)\\,dt=(dx,dy%s)$ est le déplacement élémentaire le long de la courbe ($\\ell$ désigne l'abscisse curviligne) ; $\\vec\\tau=\\gamma'/\\|\\gamma'\\|$ est le vecteur tangent unitaire." % (",dz" if dim == 3 else ""))
    _lt("$ds=\\|\\gamma'(t)\\|\\,dt$ est l'élément de longueur (norme de $d\\vec\\ell$).")


def _formule_arc(dim):
    nm = ["P", "Q", "R"][:dim]
    dd = ["dx", "dy", "dz"][:dim]
    xs = ["x", "y", "z"][:dim]
    forme = " + ".join("%s\\,%s" % (nm[i], dd[i]) for i in range(dim))
    par = " + ".join("%s(\\gamma(t))\\,%s'(t)" % (nm[i], xs[i]) for i in range(dim))
    _titre("Formule")
    _P(r"\int_\gamma \vec V\cdot d\vec\ell \;=\; \int_\gamma " + forme + r" \;=\; \int_a^b \Bigl(" + par + r"\Bigr)\,dt")
    _lt("$\\vec V=(%s)$ et $\\gamma(t)=(%s)$, $t\\in[a,b]$, paramètre l'arc. Les trois expressions sont :" % (",".join(nm), ",".join("%s(t)" % c for c in xs)))
    _lt("(1) $\\int_\\gamma \\vec V\\cdot d\\vec\\ell$ : la circulation de $\\vec V$ le long de l'arc, écrite avec les vecteurs ;")
    _lt("(2) $\\int_\\gamma %s$ : la même circulation, écrite avec la forme différentielle ;" % forme)
    _lt("(3) $\\int_a^b(%s)\\,dt$ : le calcul par paramétrage, car $\\vec V(\\gamma(t))\\cdot\\gamma'(t)=%s$." % (par, " + ".join("%s\\,%s'" % (nm[i], xs[i]) for i in range(dim))))


def _formule_green():
    _titre("Formule de Green–Riemann")
    _P(r"\oint_{\partial D^+} \vec V\cdot d\vec\ell \;=\; \oint_{\partial D^+} P\,dx+Q\,dy \;=\; \int_a^b \Bigl(P(\gamma(t))\,x'(t)+Q(\gamma(t))\,y'(t)\Bigr)\,dt \;=\; \iint_D \Bigl(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\Bigr)\,dx\,dy")
    _lt("Avec $\\vec V=(P,Q)$ et $\\gamma(t)=(x(t),y(t))$, $t\\in[a,b]$, qui paramètre $\\partial D^+$, les quatre expressions sont :")
    _lt("(1) $\\oint_{\\partial D^+}\\vec V\\cdot d\\vec\\ell$ : la circulation de $\\vec V$ sur le bord, écrite avec les vecteurs ;")
    _lt("(2) $\\oint_{\\partial D^+}P\\,dx+Q\\,dy$ : la même circulation, écrite avec la forme différentielle ;")
    _lt("(3) $\\int_a^b\\bigl(P(\\gamma(t))\\,x'(t)+Q(\\gamma(t))\\,y'(t)\\bigr)\\,dt$ : le calcul par paramétrage, car $\\vec V(\\gamma(t))\\cdot\\gamma'(t)=P\\,x'+Q\\,y'$ ;")
    _lt("(4) $\\iint_D\\bigl(\\frac{\\partial Q}{\\partial x}-\\frac{\\partial P}{\\partial y}\\bigr)\\,dx\\,dy$ : l'intégrale double sur $D$.")


def _vec(c):
    return r"\left(" + ", ".join(sp.latex(q) for q in c) + r"\right)"


def _decrit_courbe(t, a, b, comp, tnom):
    noms = "xyz"[:len(comp)]
    corps = r"\begin{cases}" + r" \\ ".join("%s(%s) = %s" % (noms[i], sp.latex(t), sp.latex(comp[i])) for i in range(len(comp))) + r"\end{cases}"
    _P(r"\gamma : [" + _borne_lat(a) + "," + _borne_lat(b) + r"] \to \mathbb{R}^{" + str(len(comp)) + r"}, \quad \gamma(" + sp.latex(t) + r") = " + _vec(comp) + r", \qquad " + corps)


def _regularite_courbe(t, a, b, comp, dc):
    fa, fb = float(sp.N(a)), float(sp.N(b))
    ok = _fini_sur(list(comp) + list(dc), [t], [(fa, fb)], 800)
    return ok and not any(c.has(sp.Abs, sp.sign, sp.Piecewise) for c in dc)


def _potentiel(F, vars_):
    """Potentiel φ avec ∇φ = F pour F sur ℝ^n (2 ou 3 composantes) ; None si introuvable (élémentaire seulement)."""
    try:
        phi = sp.integrate(F[0], vars_[0])
        if phi.has(sp.Integral) or _speciale(phi):
            return None
        for k in range(1, len(vars_)):
            reste = sp.simplify(F[k] - sp.diff(phi, vars_[k]))
            # reste ne doit dépendre que des variables d'indice >= k
            for j in range(k):
                if reste.has(vars_[j]):
                    return None
            g = sp.integrate(reste, vars_[k])
            if g.has(sp.Integral) or _speciale(g):
                return None
            phi = phi + g
        phi = sp.simplify(phi)
        for k in range(len(vars_)):
            if sp.simplify(sp.diff(phi, vars_[k]) - F[k]) != 0:
                return None
        return phi
    except Exception:
        return None


def _rot(F):
    if len(F) == 2:
        return [sp.simplify(sp.diff(F[1], X) - sp.diff(F[0], Y))]
    P_, Q_, R_ = F
    return [sp.simplify(sp.diff(R_, Y) - sp.diff(Q_, Z)),
            sp.simplify(sp.diff(P_, Z) - sp.diff(R_, X)),
            sp.simplify(sp.diff(Q_, X) - sp.diff(P_, Y))]


def _etude_arc(champ_txt, courbe, loc):
    t, a, b, comp, tnom = courbe
    dim = len(comp)
    vars_ = (X, Y, Z)[:dim]
    mode, F = _parse_champ(champ_txt, loc) if champ_txt.strip() else ("longueur", [])
    if mode == "aire" and dim != 2:
        raise _Abandon("L'aire ne se calcule que pour une courbe plane.")
    if mode == "scalaire" and champ_txt.strip().lower() in ("1", ""):
        mode = "longueur"
    if mode == "champ" and len(F) != dim:
        raise _Abandon("Le champ a %d composantes alors que la courbe vit dans R^%d." % (len(F), dim))

    dc = [sp.diff(c, t) for c in comp]
    comp_s = [sp.simplify(c) for c in comp]
    dc = [sp.simplify(c) for c in dc]

    _titre("Intégrale curviligne le long d'un arc paramétré")
    _notations(dim, "champ" if mode == "champ" else "scalaire")
    _decrit_courbe(t, a, b, comp_s, tnom)
    _P(r"\gamma'(" + sp.latex(t) + r") = " + _vec(dc))

    # régularité
    fa, fb = float(sp.N(a)), float(sp.N(b))
    if fa >= fb:
        raise _Abandon("Il faut a < b pour le paramètre.")
    if _regularite_courbe(t, a, b, comp_s, dc):
        _lt("Les composantes de $\\gamma$ sont de classe $C^1$ sur $[%s,%s]$ (fonctions élémentaires sans singularité sur cet intervalle) : $\\gamma$ est un arc paramétré $C^1$ ; le sens de parcours est celui des $%s$ croissants." % (_borne_lat(a), _borne_lat(b), sp.latex(t)))
    else:
        _lt("Attention : $\\gamma$ présente un point où une composante n'est pas $C^1$ sur $[%s,%s]$ ; il faut découper l'arc en morceaux $C^1$ (relation de Chasles)." % (_borne_lat(a), _borne_lat(b)))
    ferme = all(sp.simplify(c.subs(t, a) - c.subs(t, b)) == 0 for c in comp_s)
    if ferme:
        _lt("$\\gamma(a)=\\gamma(b)$ : l'arc est fermé.")
    r2, nv = _norme_vitesse(dc, t)

    valeur = None
    gexpr = None
    if mode == "longueur":
        gexpr = nv
        _lt("Longueur de l'arc (définition) : $L(\\gamma)=\\int_a^b \\|\\gamma'(%s)\\|\\,d%s$, avec $\\|\\gamma'\\|=\\sqrt{%s}%s$." % (sp.latex(t), sp.latex(t), sp.latex(r2), ("" if sp.latex(nv) == r"\sqrt{" + sp.latex(r2) + "}" else " = " + sp.latex(nv))))
        if nv.has(sp.Abs):
            raise _Abandon("‖γ'‖ contient une valeur absolue : découpez l'arc en morceaux où le signe est constant.")
        valeur = _ligne_integrale(gexpr, t, a, b)
        _P(r"L(\gamma) = " + sp.latex(valeur))
    elif mode == "aire":
        _lt("Pour un arc fermé simple parcouru dans le sens direct, la formule de Green–Riemann (avec $P=-\\tfrac{y}{2}$, $Q=\\tfrac{x}{2}$, donc $\\partial_xQ-\\partial_yP=1$) donne $\\mathcal{A}(D)=\\tfrac12\\oint_\\gamma (x\\,dy-y\\,dx)$.")
        if not ferme:
            raise _Abandon("L'aire par ∮ ½(x dy − y dx) exige un arc fermé.")
        gexpr = sp.simplify(sp.Rational(1, 2) * (comp_s[0] * dc[1] - comp_s[1] * dc[0]))
        _P(r"\tfrac12\,\bigl(x\,y' - y\,x'\bigr) = " + sp.latex(gexpr))
        valeur = _ligne_integrale(gexpr, t, a, b)
        if valeur.is_negative:
            _lt("La valeur est négative : $\\gamma$ parcourt le bord dans le sens horaire ; l'aire vaut l'opposé.")
            valeur = -valeur
        _P(r"\mathcal{A}(D) = " + sp.latex(valeur))
    elif mode == "scalaire":
        f = F[0]
        fg = sp.simplify(f.subs({X: comp_s[0], Y: comp_s[1]} if dim == 2 else {X: comp_s[0], Y: comp_s[1], Z: comp_s[2]}, simultaneous=True))
        _lt("Intégrale d'une fonction le long de $\\gamma$ : $\\int_\\gamma f\\,ds=\\int_a^b f(\\gamma(%s))\\,\\|\\gamma'(%s)\\|\\,d%s$." % (sp.latex(t), sp.latex(t), sp.latex(t)))
        _P(r"f(\gamma(" + sp.latex(t) + r")) = " + sp.latex(fg) + r", \qquad \|\gamma'(" + sp.latex(t) + r")\| = \sqrt{" + sp.latex(r2) + r"}" + ("" if sp.latex(nv) == r"\sqrt{" + sp.latex(r2) + "}" else " = " + sp.latex(nv)))
        if nv.has(sp.Abs):
            raise _Abandon("‖γ'‖ contient une valeur absolue : découpez l'arc en morceaux où le signe est constant.")
        gexpr = sp.simplify(fg * nv)
        valeur = _ligne_integrale(gexpr, t, a, b)
        _P(r"\int_\gamma f\,ds = " + sp.latex(valeur))
    else:
        sub = {X: comp_s[0], Y: comp_s[1]}
        if dim == 3:
            sub[Z] = comp_s[2]
        _formule_arc(dim)
        _P(r"\vec V = \left(" + ", ".join(sp.latex(c) for c in F) + r"\right)")
        Fg = [sp.simplify(c.subs(sub, simultaneous=True)) for c in F]
        terme = [sp.simplify(Fg[i] * dc[i]) for i in range(dim)]
        gexpr = sp.simplify(sum(terme))
        rot = _rot(F)
        nul = all(sp.simplify(c) == 0 for c in rot)
        lisse = nul and _lisse_global(F, vars_)
        phi = _potentiel(F, vars_) if lisse else None
        if ferme and lisse:
            # résultat prévisible : justification brève, aucun calcul d'intégrale
            _titre("Résultat prévisible")
            if dim == 2:
                _P(r"\partial_x Q - \partial_y P = " + sp.latex(sp.diff(F[1], X)) + " - " + _par(sp.diff(F[0], Y)) + " = " + sp.latex(rot[0]))
            else:
                _P(r"\operatorname{rot}\vec V = \left(" + ", ".join(sp.latex(c) for c in rot) + r"\right)")
            _lt("$\\vec V$ est de classe $C^1$ sur $\\mathbb{R}^%d$, ensemble étoilé, et son rotationnel est nul : d'après le lemme de Poincaré, $\\vec V=\\nabla\\varphi$ pour un potentiel $\\varphi$. L'arc est fermé, donc $\\gamma(b)=\\gamma(a)$ et le théorème fondamental pour les champs de gradient donne $\\int_\\gamma \\vec V\\cdot d\\vec\\ell=\\varphi(\\gamma(b))-\\varphi(\\gamma(a))=0$." % dim)
            if phi is not None:
                _P(r"\varphi(" + ",".join(sp.latex(v) for v in vars_) + r") = " + sp.latex(phi) + r" \qquad (\nabla\varphi=\vec V\ \text{vérifié})")
            valeur = sp.Integer(0)
            _P(r"\int_\gamma \vec V\cdot d\vec\ell = 0")
        elif lisse and phi is not None:
            _titre("Calcul par le potentiel")
            if dim == 2:
                _P(r"\partial_x Q - \partial_y P = " + sp.latex(sp.diff(F[1], X)) + " - " + _par(sp.diff(F[0], Y)) + " = " + sp.latex(rot[0]))
            else:
                _P(r"\operatorname{rot}\vec V = \left(" + ", ".join(sp.latex(c) for c in rot) + r"\right)")
            _lt("Le rotationnel est nul et $\\vec V$ est de classe $C^1$ sur $\\mathbb{R}^%d$, ensemble étoilé : lemme de Poincaré, $\\vec V=\\nabla\\varphi$. Une primitive $\\varphi$ évite d'intégrer le long de $\\gamma$ : $\\int_\\gamma\\vec V\\cdot d\\vec\\ell=\\varphi(\\gamma(b))-\\varphi(\\gamma(a))$ (théorème fondamental pour un champ de gradient)." % dim)
            _P(r"\varphi(" + ",".join(sp.latex(v) for v in vars_) + r") = " + sp.latex(phi) + r" \qquad (\nabla\varphi=\vec V\ \text{vérifié})")
            A_ = [c.subs(t, a) for c in comp_s]
            B_ = [c.subs(t, b) for c in comp_s]
            _P(r"\gamma(a) = " + _vec(A_) + r", \qquad \gamma(b) = " + _vec(B_))
            pa = sp.simplify(phi.subs(dict(zip(vars_, A_)), simultaneous=True))
            pb = sp.simplify(phi.subs(dict(zip(vars_, B_)), simultaneous=True))
            valeur = sp.simplify(pb - pa)
            _P(r"\begin{aligned} \int_\gamma \vec V\cdot d\vec\ell &= \varphi(\gamma(b)) - \varphi(\gamma(a)) && " + _T("(théorème fondamental pour un champ de gradient)") +
               r" \\ &= " + sp.latex(pb) + " - " + (r"\left(" + sp.latex(pa) + r"\right)" if (pa.is_Add or pa.could_extract_minus_sign()) else sp.latex(pa)) + r" \\ &= " + sp.latex(valeur) + r" \end{aligned}")
        else:
            _P(r"\vec V(\gamma(" + sp.latex(t) + r")) = \left(" + ", ".join(sp.latex(c) for c in Fg) + r"\right), \qquad \gamma'(" + sp.latex(t) + r") = " + _vec(dc))
            _P(r"\vec V(\gamma(" + sp.latex(t) + r"))\cdot\gamma'(" + sp.latex(t) + r") = " + " + ".join(sp.latex(c) for c in terme) + " = " + sp.latex(gexpr))
            valeur = _ligne_integrale(gexpr, t, a, b)
            _P(r"\int_\gamma \vec V\cdot d\vec\ell = " + sp.latex(valeur))

            _titre("Le champ dérive-t-il d'un potentiel ?")
            if dim == 2:
                _P(r"\partial_x Q - \partial_y P = " + sp.latex(sp.diff(F[1], X)) + " - " + _par(sp.diff(F[0], Y)) + " = " + sp.latex(rot[0]))
            else:
                _P(r"\operatorname{rot}\vec V = \left(" + ", ".join(sp.latex(c) for c in rot) + r"\right)")
            if not nul:
                _lt("Le rotationnel n'est pas nul : $\\vec V$ ne dérive d'aucun potentiel, donc la circulation dépend du chemin joignant les extrémités%s." % (" ; sur un arc fermé, rien n'impose une circulation nulle" if ferme else ""))
            elif lisse:
                _lt("Le rotationnel est nul et $\\vec V$ est de classe $C^1$ sur $\\mathbb{R}^%d$, ensemble étoilé : lemme de Poincaré, $\\vec V$ dérive d'un potentiel $\\varphi$ avec $\\nabla\\varphi=\\vec V$." % dim)
                if phi is not None:
                    _P(r"\varphi(" + ",".join(sp.latex(v) for v in vars_) + r") = " + sp.latex(phi) + r" \qquad (\nabla\varphi=\vec V\ \text{vérifié})")
                    A_ = [c.subs(t, a) for c in comp_s]
                    B_ = [c.subs(t, b) for c in comp_s]
                    pa = sp.simplify(phi.subs(dict(zip(vars_, A_)), simultaneous=True))
                    pb = sp.simplify(phi.subs(dict(zip(vars_, B_)), simultaneous=True))
                    bloc = (r"\begin{aligned} \int_\gamma \vec V\cdot d\vec\ell &= \varphi(\gamma(b)) - \varphi(\gamma(a)) && " + _T("(théorème fondamental pour un champ de gradient)") +
                            r" \\ &= " + sp.latex(pb) + " - " + (r"\left(" + sp.latex(pa) + r"\right)" if (pa.is_Add or pa.could_extract_minus_sign()) else sp.latex(pa)) + r" \\ &= " + sp.latex(sp.simplify(pb - pa)) + r" \end{aligned}")
                    _P(bloc)
                    if sp.simplify(pb - pa - valeur) == 0:
                        _lt("La valeur obtenue avec le potentiel coïncide avec le calcul direct par paramétrage.")
                    else:
                        _lt("Écart avec le calcul direct : à vérifier.")
            else:
                _lt("Le rotationnel est nul, mais $\\vec V$ n'est pas définie et $C^1$ sur tout $\\mathbb{R}^%d$ : le domaine n'est pas simplement connexe en général ; on ne peut pas conclure à l'existence d'un potentiel. La circulation sur un arc fermé peut être non nulle, comme le calcul direct ci-dessus le montre." % dim)
    # graphe
    try:
        _graphe_courbe(t, a, b, comp_s, F if (mode == "champ" and dim == 2) else None)
    except Exception:
        plt.close("all")
    # contrôle numérique
    if gexpr is not None:
        vn = _num_int(gexpr, t, a, b)
        if vn is not None and valeur is not None:
            vx = _L._num(valeur)
            if vx is not None:
                _P(_T("Contrôle numérique : l'intégrale calculée par quadrature vaut environ $%s$, la valeur exacte vaut environ $%s$." % (_fmt_num(vn), _fmt_num(vx))))
    return valeur


# -------------------------------------------------------------
# Green–Riemann
# -------------------------------------------------------------
def _borne_y(cons, u, v):
    """Bornes de la variable intérieure v en fonction de u : (bornes inf, bornes sup, contraintes sur u seul)."""
    inf, sup, seul = [], [], []
    for c in cons:
        c = sp.expand(c)
        if not c.has(v):
            seul.append(c)
            continue
        p = sp.Poly(c, v)
        d = p.degree()
        if d == 1:
            al = p.coeff_monomial(v)
            be = sp.expand(c - al * v)
            if al.free_symbols:
                raise _Abandon("coefficient non constant")
            borne = sp.simplify(-be / al)
            (inf if al > 0 else sup).append(borne)
        elif d == 2:
            a2 = p.coeff_monomial(v ** 2)
            if a2.free_symbols or a2 >= 0:
                raise _Abandon("contrainte non convexe en la variable intérieure")
            racines = sp.solve(sp.Eq(c, 0), v)
            if len(racines) != 2:
                raise _Abandon("racines")
            r = [sp.simplify(q) for q in racines]
            # ordre : r0 < r1 sur un point d'essai
            pt = {u: sp.Rational(1, 7)}
            try:
                if float(sp.N(r[0].subs(pt))) > float(sp.N(r[1].subs(pt))):
                    r = r[::-1]
            except Exception:
                pass
            inf.append(r[0])
            sup.append(r[1])
        else:
            raise _Abandon("degré")
    return inf, sup, seul


def _solu_unique(liste, comp, u, a, b, cmp_sup):
    """Choisit parmi les bornes celle qui domine sur [a,b] (max des inf / min des sup) ; None si indécis."""
    if len(liste) == 1:
        return liste[0]
    ts = [float(a) + (float(b) - float(a)) * k / 12 for k in range(1, 12)]
    for cand in liste:
        ok = True
        for other in liste:
            if other is cand:
                continue
            for s in ts:
                try:
                    vc = float(sp.N(cand.subs(u, s)))
                    vo = float(sp.N(other.subs(u, s)))
                except Exception:
                    return None
                if (cmp_sup and vc > vo + 1e-9) or ((not cmp_sup) and vc < vo - 1e-9):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return cand
    return None


def _type_simple(cons, u, v):
    """Domaine {a<=u<=b, g1(u)<=v<=g2(u)} ; None si non reconnu."""
    try:
        inf, sup, seul = _borne_y(cons, u, v)
    except _Abandon:
        return None
    except Exception:
        return None
    if not inf or not sup:
        return None
    lo_u, hi_u = [], []
    for c in seul:
        c = sp.expand(c)
        if not c.has(u):
            continue
        p = sp.Poly(c, u)
        if p.degree() != 1:
            return None
        al = p.coeff_monomial(u)
        be = sp.expand(c - al * u)
        if al.free_symbols or be.free_symbols:
            return None
        (lo_u if al > 0 else hi_u).append(sp.nsimplify(-be / al))
    # intervalle de u
    def pts_croisement():
        res = []
        for gi in inf:
            for gs in sup:
                try:
                    rac = sp.solve(sp.Eq(gi, gs), u)
                except Exception:
                    rac = []
                for r in rac:
                    if r.is_real:
                        res.append(sp.simplify(r))
        return res
    cand_lo = list(lo_u)
    cand_hi = list(hi_u)
    if not cand_lo or not cand_hi:
        croix = sorted(set(pts_croisement()), key=lambda q: float(sp.N(q)))
        if not cand_lo and not cand_hi:
            if len(croix) < 2:
                return None
            for k in range(len(croix) - 1):
                m = (float(sp.N(croix[k])) + float(sp.N(croix[k + 1]))) / 2
                if all(float(sp.N(gs.subs(u, m))) >= float(sp.N(gi.subs(u, m))) for gi in inf for gs in sup):
                    cand_lo, cand_hi = [croix[k]], [croix[k + 1]]
                    break
        elif not cand_lo:
            b0 = max(float(sp.N(h)) for h in cand_hi)
            c = [q for q in croix if float(sp.N(q)) < b0 - 1e-9]
            if not c:
                return None
            cand_lo = [max(c, key=lambda q: float(sp.N(q)))]
        else:
            a0 = min(float(sp.N(h)) for h in cand_lo)
            c = [q for q in croix if float(sp.N(q)) > a0 + 1e-9]
            if not c:
                return None
            cand_hi = [min(c, key=lambda q: float(sp.N(q)))]
    if not cand_lo or not cand_hi:
        return None
    a = max(cand_lo, key=lambda q: float(sp.N(q)))
    b = min(cand_hi, key=lambda q: float(sp.N(q)))
    if float(sp.N(a)) >= float(sp.N(b)):
        return None
    g1 = _solu_unique(inf, None, u, sp.N(a), sp.N(b), False)
    g2 = _solu_unique(sup, None, u, sp.N(a), sp.N(b), True)
    if g1 is None or g2 is None:
        return None
    # vérification g1 <= g2 sur ]a,b[
    for k in range(1, 20):
        s = float(sp.N(a)) + (float(sp.N(b)) - float(sp.N(a))) * k / 20
        if float(sp.N(g2.subs(u, s))) < float(sp.N(g1.subs(u, s))) - 1e-9:
            return None
    return a, b, sp.simplify(g1), sp.simplify(g2)


def _ellipse(cons):
    """Une seule contrainte : ellipse pleine. Renvoie (x0, y0, A, B) ou None."""
    if len(cons) != 1:
        return None
    c = sp.expand(cons[0])
    try:
        p = sp.Poly(c, X, Y)
    except Exception:
        return None
    if p.total_degree() != 2:
        return None
    cxx, cyy, cxy = p.coeff_monomial(X ** 2), p.coeff_monomial(Y ** 2), p.coeff_monomial(X * Y)
    cx, cy, c0 = p.coeff_monomial(X), p.coeff_monomial(Y), p.coeff_monomial(1)
    if cxy != 0 or not (cxx < 0 and cyy < 0):
        return None
    x0 = -cx / (2 * cxx)
    y0 = -cy / (2 * cyy)
    K = -(c0 - cxx * x0 ** 2 - cyy * y0 ** 2)
    K = sp.simplify(K)
    if not K.is_positive:
        return None
    A = sp.simplify(sp.sqrt(K / (-cxx)))
    B = sp.simplify(sp.sqrt(K / (-cyy)))
    return sp.nsimplify(x0), sp.nsimplify(y0), A, B


_TXT_DOM = []


def _desc_domaine_lat(cons):
    if _TXT_DOM:
        return r",\ ".join(_TXT_DOM)
    morceaux = []
    for c in cons:
        morceaux.append(sp.latex(sp.simplify(c)) + r" \geq 0")
    return r",\ ".join(morceaux)


def _frontiere_type(a, b, g1, g2, u, v, u_est_x):
    """Pièces du bord orienté positivement. Chaque pièce : dict(nom, x, y, t, t0, t1, signe)."""
    t = sp.Symbol("t", real=True)
    sub = lambda e: e.subs(u, t)
    pieces = []
    if u_est_x:
        # bas, droite, haut, gauche
        pieces.append(dict(nom="bord inférieur $y=%s$ (parcouru dans le sens des $x$ croissants)" % sp.latex(g1), x=t, y=sub(g1), t=t, t0=a, t1=b, signe=1))
        if sp.simplify(g2.subs(u, b) - g1.subs(u, b)) != 0:
            pieces.append(dict(nom="côté droit $x=%s$ (de bas en haut)" % sp.latex(b), x=b, y=t, t=t, t0=g1.subs(u, b), t1=g2.subs(u, b), signe=1))
        pieces.append(dict(nom="bord supérieur $y=%s$ (parcouru dans le sens des $x$ décroissants)" % sp.latex(g2), x=t, y=sub(g2), t=t, t0=a, t1=b, signe=-1))
        if sp.simplify(g2.subs(u, a) - g1.subs(u, a)) != 0:
            pieces.append(dict(nom="côté gauche $x=%s$ (de haut en bas)" % sp.latex(a), x=a, y=t, t=t, t0=g1.subs(u, a), t1=g2.subs(u, a), signe=-1))
    else:
        # u = y : droite (x=g2(y)) montante, haut (y=b) vers la gauche, gauche (x=g1(y)) descendante, bas (y=a) vers la droite
        pieces.append(dict(nom="bord droit $x=%s$ (de bas en haut)" % sp.latex(g2), x=sub(g2), y=t, t=t, t0=a, t1=b, signe=1))
        if sp.simplify(g2.subs(u, b) - g1.subs(u, b)) != 0:
            pieces.append(dict(nom="côté haut $y=%s$ (de droite à gauche)" % sp.latex(b), x=t, y=b, t=t, t0=g1.subs(u, b), t1=g2.subs(u, b), signe=-1))
        pieces.append(dict(nom="bord gauche $x=%s$ (de haut en bas)" % sp.latex(g1), x=sub(g1), y=t, t=t, t0=a, t1=b, signe=-1))
        if sp.simplify(g2.subs(u, a) - g1.subs(u, a)) != 0:
            pieces.append(dict(nom="côté bas $y=%s$ (de gauche à droite)" % sp.latex(a), x=t, y=a, t=t, t0=g1.subs(u, a), t1=g2.subs(u, a), signe=1))
    return pieces


def _courbes_numeriques(pieces):
    res = []
    for pc in pieces:
        t = pc["t"]
        ts = np.linspace(float(sp.N(pc["t0"])), float(sp.N(pc["t1"])), 120)
        if pc["signe"] == -1:
            ts = ts[::-1]
        fx = sp.lambdify(t, pc["x"], "numpy")
        fy = sp.lambdify(t, pc["y"], "numpy")
        res.append((np.asarray(fx(ts), dtype=float) * np.ones_like(ts), np.asarray(fy(ts), dtype=float) * np.ones_like(ts)))
    return res


def _etude_green(champ_txt, dom_txt, loc):
    cons = _parse_domaine(dom_txt, loc)
    mode, F = _parse_champ(champ_txt, loc) if champ_txt.strip() else ("aire", [])
    if mode == "aire":
        P_, Q_ = -Y / 2, X / 2
        aire = True
    elif mode == "champ" and len(F) == 2:
        P_, Q_ = F
        aire = False
    else:
        raise _Abandon("Green–Riemann : donnez un champ plan à deux composantes séparées par une virgule (par exemple « -y, x »), ou le mot « aire ».")
    rot = sp.simplify(sp.diff(Q_, X) - sp.diff(P_, Y))

    _titre("Théorème de Green–Riemann")
    _notations(2, "domaine")
    _titre("Données")
    _P(r"D=\{(x,y)\in\mathbb{R}^2 : " + _desc_domaine_lat(cons) + r"\}")
    if aire:
        _lt("Calcul de l'aire : on choisit $P=-\\tfrac{y}{2}$ et $Q=\\tfrac{x}{2}$, de sorte que $\\partial_xQ-\\partial_yP=1$.")
    else:
        _P(r"\vec V(x,y) = \bigl(P(x,y),\,Q(x,y)\bigr), \qquad P(x,y) = " + sp.latex(P_) + r", \qquad Q(x,y) = " + sp.latex(Q_))

    # description du bord et du type de domaine
    ell = _ellipse(cons)
    pieces = []
    type_ = None
    if ell is not None:
        x0, y0, A, B = ell
        type_ = "ellipse"
    else:
        for (u, v, ux) in ((X, Y, True), (Y, X, False)):
            r = _type_simple(cons, u, v)
            if r is not None:
                a, b, g1, g2 = r
                type_ = ("simple", u, v, ux, a, b, g1, g2)
                break
    if type_ is None:
        raise _Abandon("La forme du domaine n'est pas reconnue (disque/ellipse, ou domaine entre deux courbes a<=x<=b, g1(x)<=y<=g2(x), ou son symétrique en y).")

    # hypothèses
    xs_box = None
    if type_ == "ellipse":
        bx = (float(sp.N(x0 - A)), float(sp.N(x0 + A)))
        by = (float(sp.N(y0 - B)), float(sp.N(y0 + B)))
    else:
        _, u, v, ux, a, b, g1, g2 = type_
        ua, ub = float(sp.N(a)), float(sp.N(b))
        samp = [ua + (ub - ua) * k / 40 for k in range(41)]
        vv = []
        for s in samp:
            vv.append(float(sp.N(g1.subs(u, s))))
            vv.append(float(sp.N(g2.subs(u, s))))
        if ux:
            bx, by = (ua, ub), (min(vv), max(vv))
        else:
            by, bx = (ua, ub), (min(vv), max(vv))
    comp_reg = [P_, Q_, sp.diff(P_, X), sp.diff(P_, Y), sp.diff(Q_, X), sp.diff(Q_, Y)]
    reg = _fini_sur(comp_reg, (X, Y), [bx, by], 1500) and not any(c.has(sp.Abs, sp.sign, sp.Piecewise) for c in comp_reg)
    _lt("Hypothèses : $D$ est compact (fermé et borné), son bord $\\partial D$ est une courbe fermée simple, $C^1$ par morceaux, ici orientée positivement (le domaine reste à gauche).")
    if reg:
        _lt("Les fonctions $P$ et $Q$ sont de classe $C^1$ sur un voisinage de $D$ (expressions élémentaires sans singularité sur cette région) : le théorème de Green–Riemann s'applique.")
        pass
    else:
        _lt("Les fonctions $P$ ou $Q$ présentent une singularité sur $D$ ou son bord : l'hypothèse $C^1$ sur un voisinage de $D$ n'est pas satisfaite, Green–Riemann ne peut pas être appliqué tel quel. Seul le calcul direct de la circulation sur le bord est valable.")
    _formule_green()

    # --- bord
    t = sp.Symbol("t", real=True)
    if type_ == "ellipse":
        th = t
        pc = dict(nom="le bord est l'ellipse paramétrée par $(x,y)=(%s + %s\\cos t,\\ %s + %s\\sin t)$, $t\\in[0,2\\pi]$ (sens trigonométrique)" % (sp.latex(x0), sp.latex(A), sp.latex(y0), sp.latex(B)),
                  x=x0 + A * sp.cos(t), y=y0 + B * sp.sin(t), t=t, t0=0, t1=2 * sp.pi, signe=1)
        pieces = [pc]
    else:
        pieces = _frontiere_type(a, b, g1, g2, u, v, ux)

    if reg and not aire and sp.simplify(rot) == 0:
        _titre("Résultat prévisible")
        gch = sp.latex(sp.diff(Q_, X)) + " - " + _par(sp.diff(P_, Y))
        _P(r"\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} = " + gch + " = 0")
        _lt("$P$ et $Q$ sont de classe $C^1$ sur un voisinage du compact $D$ et $\\partial_xQ-\\partial_yP=0$ sur $D$ : d'après Green–Riemann, la circulation sur le bord vaut $\\iint_D 0\\,dx\\,dy=0$, sans aucun calcul sur le bord.")
        _P(r"\oint_{\partial D^+} \vec V\cdot d\vec\ell = \oint_{\partial D^+} P\,dx+Q\,dy = \iint_D 0\,dx\,dy = 0")
        try:
            _graphe_domaine(_courbes_numeriques(pieces), (P_, Q_))
        except Exception:
            plt.close("all")
        return

    # --- Membre de gauche
    _titre("Membre de gauche : circulation sur le bord")
    total_g = 0
    ok_g = True
    for k, pc in enumerate(pieces):
        xt, yt, tt = pc["x"], pc["y"], pc["t"]
        sub = {X: xt, Y: yt}
        g = sp.simplify(P_.subs(sub, simultaneous=True) * sp.diff(xt, tt) + Q_.subs(sub, simultaneous=True) * sp.diff(yt, tt))
        _lt("Morceau %d : %s." % (k + 1, pc["nom"]))
        _P(r"\gamma_{%d}(t) = \bigl(%s,\ %s\bigr),\quad t\in[%s,%s],\qquad \gamma_{%d}'(t) = \bigl(%s,\ %s\bigr)" % (k + 1, sp.latex(xt), sp.latex(yt), _borne_lat(pc["t0"]), _borne_lat(pc["t1"]), k + 1, sp.latex(sp.diff(xt, tt)), sp.latex(sp.diff(yt, tt))))
        _P(r"P\,x' + Q\,y' = " + sp.latex(g))
        if not reg and not g.is_finite:
            pass
        try:
            v_ = _ligne_integrale(g, tt, pc["t0"], pc["t1"])
        except _Abandon:
            ok_g = False
            continue
        if pc["signe"] == -1:
            _lt("Ce morceau est parcouru dans le sens opposé à celui du paramètre : on prend l'opposé.")
            v_ = -v_
            _P(r"\int_{C_%d} P\,dx+Q\,dy = " % (k + 1) + sp.latex(v_))
        else:
            _P(r"\int_{C_%d} P\,dx+Q\,dy = " % (k + 1) + sp.latex(v_))
        total_g = total_g + v_
    if ok_g:
        total_g = sp.simplify(total_g)
        if len(pieces) > 1:
            somme = " + ".join("I_%d" % (i + 1) for i in range(len(pieces)))
            _P(r"\oint_{\partial D^+} P\,dx+Q\,dy = " + somme + " = " + sp.latex(total_g) + r" \qquad (\text{relation de Chasles})")
        else:
            _P(r"\oint_{\partial D^+} P\,dx+Q\,dy = " + sp.latex(total_g))
    else:
        total_g = None

    # --- Membre de droite
    total_d = None
    iter_txt = None
    if reg:
        _titre("Membre de droite : intégrale double")
        gauche_ = sp.latex(sp.diff(Q_, X)) + " - " + _par(sp.diff(P_, Y))
        _P(r"\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} = " + gauche_ + ("" if gauche_.replace(" ", "") == sp.latex(rot).replace(" ", "") else " = " + sp.latex(rot)))
        try:
            if type_ == "ellipse":
                r_, th = sp.symbols("r theta", positive=True)
                f_pol = sp.simplify(rot.subs({X: x0 + A * r_ * sp.cos(th), Y: y0 + B * r_ * sp.sin(th)}, simultaneous=True) * A * B * r_)
                _lt("Changement de variables elliptiques $x=%s+%s\\,r\\cos\\theta$, $y=%s+%s\\,r\\sin\\theta$ : application $C^1$, bijective de $]0,1]\\times]0,2\\pi[$ sur $D$ privé d'un ensemble de mesure nulle, de jacobien $%s\\,r>0$. L'intégrande rot $\\cdot\\,|J|$ est continue sur le pavé $[0,1]\\times[0,2\\pi]$ : le théorème de Fubini s'applique." % (sp.latex(x0), sp.latex(A), sp.latex(y0), sp.latex(B), sp.latex(A * B)))
                iter_txt = (r"\int_0^{2\pi}\!\!\int_0^{1}" + sp.latex(f_pol) + r"\,dr\,d\theta", "changement de variables elliptiques puis Fubini")
                _P(r"\iint_D (\partial_xQ-\partial_yP)\,dx\,dy = " + iter_txt[0])
                _lt("Intégrale intérieure en $r$ ($\\theta$ fixé) :")
                F_in = _simp(sp.integrate(f_pol, (r_, 0, 1)))
                _ligne_integrale(f_pol, r_, 0, 1, F_in)
                _lt("Intégrale extérieure en $\\theta$ :")
                total_d = _ligne_integrale(sp.simplify(F_in), th, 0, 2 * sp.pi)
            else:
                dom_u = (u, a, b)
                inner = rot
                _lt("La fonction $\\partial_xQ-\\partial_yP$ est continue sur le compact $D$, donc intégrable ; $D$ est de la forme %s : le théorème de Fubini donne l'intégrale double comme intégrale itérée." % (
                    ("$\\{a\\le x\\le b,\\ g_1(x)\\le y\\le g_2(x)\\}$" if ux else "$\\{c\\le y\\le d,\\ h_1(y)\\le x\\le h_2(y)\\}$")))
                iter_txt = (r"\int_{" + _borne_lat(a) + "}^{" + _borne_lat(b) + r"}\!\!\int_{" + sp.latex(g1) + "}^{" + sp.latex(g2) + "}" + _par_add(inner) + r"\,d" + sp.latex(v) + r"\,d" + sp.latex(u), "Fubini")
                _P(r"\iint_D (\partial_xQ-\partial_yP)\,dx\,dy = " + iter_txt[0])
                F_in = _simp(sp.integrate(inner, (v, g1, g2)))
                _lt("Intégrale intérieure en $%s$ ($%s$ fixé) :" % (sp.latex(v), sp.latex(u)))
                _ligne_integrale(inner, v, g1, g2, F_in)
                _lt("Intégrale extérieure en $%s$ :" % sp.latex(u))
                total_d = _ligne_integrale(sp.simplify(F_in), u, a, b)
        except _Abandon as e:
            _rouge(str(e))
            total_d = None

    # --- Conclusion
    _titre("Conclusion")
    if aire:
        if total_d is not None:
            _P(r"\mathcal{A}(D) = \iint_D 1\,dx\,dy = \oint_{\partial D^+}\tfrac12(x\,dy-y\,dx) = " + sp.latex(total_d))
    elif total_g is not None and total_d is not None:
        if sp.simplify(total_g - total_d) == 0:
            somme = " + ".join("I_%d" % (i + 1) for i in range(len(pieces)))
            lignes = [r"\oint_{\partial D^+} \vec V\cdot d\vec\ell &= \oint_{\partial D^+} P\,dx+Q\,dy && " + _T("(en coordonnées)"),
                      r"&= " + (somme + " = " if len(pieces) > 1 else "") + sp.latex(total_g) + r" && " + _T("(paramétrage du bord" + (", relation de Chasles" if len(pieces) > 1 else "") + ")"),
                      r"&= \iint_D \Bigl(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\Bigr)\,dx\,dy && " + _T("(Green–Riemann)")]
            if iter_txt is not None:
                lignes.append(r"&= " + iter_txt[0] + r" && " + _T("(" + iter_txt[1] + ")"))
            lignes.append(r"&= " + sp.latex(total_d) + r" && " + _T("(calcul ci-dessus) $\\checkmark$"))
            _P(r"\begin{aligned} " + r" \\ ".join(lignes) + r" \end{aligned}")
        else:
            _lt("Les deux membres diffèrent : une erreur de signe ou de domaine est à rechercher.")
    elif total_g is not None and not reg:
        rot_nul = sp.simplify(rot) == 0
        if rot_nul and sp.simplify(total_g) != 0:
            _lt("Le rotationnel est nul partout où il est défini, pourtant la circulation sur le bord vaut $%s\\neq0$ : la singularité de $\\vec V$ dans $D$ empêche d'appliquer Green–Riemann (et $D$ privé de la singularité n'est pas simplement connexe)." % sp.latex(total_g))
        else:
            _P(r"\oint_{\partial D^+} P\,dx+Q\,dy = " + sp.latex(total_g))
    # graphe
    try:
        _graphe_domaine(_courbes_numeriques(pieces), (P_, Q_) if not aire else None)
    except Exception:
        plt.close("all")
    # contrôle numérique de l'intégrale double
    try:
        if total_d is not None:
            vx = _L._num(total_d)
            if vx is not None:
                from scipy.integrate import dblquad
                f2 = sp.lambdify((X, Y), rot, "numpy")
                if type_ == "ellipse":
                    val, _e = dblquad(lambda yy, xx: float(f2(xx, yy)), float(sp.N(x0 - A)), float(sp.N(x0 + A)),
                                      lambda xx: float(sp.N(y0 - B * sp.sqrt(max(0.0, 1 - ((xx - float(sp.N(x0))) / float(sp.N(A))) ** 2)))),
                                      lambda xx: float(sp.N(y0 + B * sp.sqrt(max(0.0, 1 - ((xx - float(sp.N(x0))) / float(sp.N(A))) ** 2)))))
                else:
                    g1f = sp.lambdify(u, g1, "numpy")
                    g2f = sp.lambdify(u, g2, "numpy")
                    if ux:
                        val, _e = dblquad(lambda yy, xx: float(f2(xx, yy)), float(sp.N(a)), float(sp.N(b)), lambda xx: float(g1f(xx)), lambda xx: float(g2f(xx)))
                    else:
                        val, _e = dblquad(lambda xx, yy: float(f2(xx, yy)), float(sp.N(a)), float(sp.N(b)), lambda yy: float(g1f(yy)), lambda yy: float(g2f(yy)))
                _P(_T("Contrôle numérique de l'intégrale double : $%s$ (valeur exacte $\\approx %s$)." % (_fmt_num(val), _fmt_num(vx))))
    except Exception:
        pass


# -------------------------------------------------------------
# Point d'entrée
# -------------------------------------------------------------
def etude_curviligne(champ_txt, courbe_txt, loc, N=None):
    """champ_txt : « P, Q » (ou « P, Q, R »), « f » (intégrale de fonction), « longueur », « aire » ;
    courbe_txt : « x=cos(t), y=sin(t), t in [0,2*pi] » (arc) ou « x^2+y^2<=1 » (domaine, Green–Riemann)."""
    _push()
    try:
        _L._VUES.clear()
        _L._GAUSS_Q.clear()
    except Exception:
        pass
    try:
        if not (courbe_txt or "").strip():
            _rouge("Indiquez la courbe (x=..., y=..., t in [a,b]) ou le domaine D (inégalités) pour appliquer Green–Riemann.")
            return
        courbe = _parse_courbe(courbe_txt, loc)
        if courbe is not None:
            _etude_arc(champ_txt or "", courbe, loc)
        else:
            _etude_green(champ_txt or "", courbe_txt, loc)
    except _Abandon as e:
        _rouge(str(e))
    except Exception as e:
        _rouge("Cas non traité par ce module (%s : %s)." % (type(e).__name__, str(e)[:120]))
