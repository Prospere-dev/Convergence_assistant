# =============================================================
# lebesgue.py : intégrale de Lebesgue, intégrales propres, impropres,
#               généralisées (programme de L2)
#   - sorties via st.latex()/st.pyplot() comme le reste de l'application
# =============================================================
import numpy as np
import sympy as sp
from sympy import S
from sympy.calculus.util import continuous_domain
import mpmath as mp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st = None  # renseigné par app.py : lebesgue.st = st

_VUES = set()


# -------------------------------------------------------------
# Outils généraux
# -------------------------------------------------------------
def _avec_delai(fn, secondes=12):
    """Exécute fn() avec un délai maximal (compatible threads). None si échec ou dépassement."""
    import threading
    import ctypes
    ident = threading.get_ident()
    fini = threading.Event()

    def _stop():
        if not fini.is_set():
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(ident), ctypes.py_object(KeyboardInterrupt))

    minuteur = threading.Timer(secondes, _stop)
    minuteur.daemon = True
    minuteur.start()
    res = None
    try:
        try:
            res = fn()
        except BaseException:
            res = None
        finally:
            fini.set()
            minuteur.cancel()
    except BaseException:
        res = None
    return res


def _P(ligne):
    """Affiche une ligne LaTeX une seule fois (aucune redondance)."""
    ligne = ligne.replace(r"\text{for}\:", r"\text{si}\:").replace(r"\text{otherwise}", r"\text{sinon}").replace(r"\log{", r"\ln{").replace(r"\operatorname{atan}", r"\arctan").replace(r"\operatorname{asin}", r"\arcsin")
    if ligne in _VUES:
        return
    _VUES.add(ligne)
    st.latex(ligne)


def _titre(s):
    _P(r"\textbf{" + s + r"}")


def _rouge(msg):
    safe = (msg or "").replace("\\", r"\textbackslash ").replace("_", r"\_")
    st.latex(r"\textcolor{red}{\text{" + safe + r"}}")


def _num(e):
    """Valeur flottante d'une expression numérique (ou None)."""
    try:
        v = complex(sp.N(e, 30))
        if abs(v.imag) < 1e-12:
            return v.real
        return None
    except Exception:
        return None


def _nul(e):
    try:
        e = sp.simplify(e)
        if e == 0:
            return True
        v = _num(e)
        return v is not None and abs(v) < 1e-12
    except Exception:
        return False


def _lat(e):
    """LaTeX d'une expression (avec \\infty)."""
    if e is None:
        return r"\text{?}"
    if e == sp.oo:
        return r"+\infty"
    if e == -sp.oo:
        return r"-\infty"
    return sp.latex(e)


def _par(e):
    s = _lat(e)
    v = _num(e) if not (e in (sp.oo, -sp.oo)) else None
    if (v is not None and v < 0) or (e == -sp.oo):
        return r"\left(" + s + r"\right)"
    if getattr(e, "is_Add", False):
        return r"\left(" + s + r"\right)"
    return s


def _exact_approx(e, chiffres=8):
    """LaTeX 'valeur exacte ≈ valeur décimale' (la décimale n'est ajoutée que pour un irrationnel)."""
    try:
        if e in (sp.oo, -sp.oo):
            return _lat(e)
        s = sp.latex(e)
        v = _num(e)
        if v is None:
            return s
        if e.is_Rational:
            return s
        return s + r"\approx " + sp.latex(sp.Float(v, chiffres))
    except Exception:
        return sp.latex(e)


def _borne_lat(b):
    if b == sp.oo:
        return r"+\infty"
    if b == -sp.oo:
        return r"-\infty"
    return sp.latex(b)


def _iv(lo, hi, ouvert_lo=True, ouvert_hi=True):
    g = r"\left]" if ouvert_lo else r"\left["
    d = r"\right[" if ouvert_hi else r"\right]"
    return g + _borne_lat(lo) + r",\," + _borne_lat(hi) + d


def _classe_limite(r):
    """('fin', valeur) | ('inf', +-oo) | ('osc', None) | ('inc', None)."""
    if r is None:
        return "inc", None
    try:
        if r.has(sp.AccumBounds) or r is sp.nan or r.has(sp.nan):
            return "osc", None
        if r in (sp.oo, -sp.oo):
            return "inf", r
        if r == sp.zoo:
            return "inf", sp.oo
        if r.is_finite is False:
            return "inf", r
        if isinstance(r, sp.Limit):
            return "inc", None
        return "fin", r
    except Exception:
        return "inc", None


def _lim(expr, x, pt, sens=None, delai=12):
    """Limite en pt (sens '+', '-' ou None) avec délai. Renvoie (classe, valeur)."""
    if sens is not None and expr.has(sp.Piecewise) and pt not in (sp.oo, -sp.oo):
        try:
            eps = sp.Rational(1, 10 ** 7) * (1 if sens == "+" else -1)
            def _branche(pw):
                for e_, c_ in pw.args:
                    try:
                        if bool(c_.subs(x, pt + eps)):
                            return e_
                    except Exception:
                        pass
                return pw
            expr = expr.replace(lambda e: isinstance(e, sp.Piecewise), _branche)
        except Exception:
            pass
    if sens is None:
        r = _avec_delai(lambda: sp.limit(expr, x, pt), delai)
    else:
        r = _avec_delai(lambda: sp.limit(expr, x, pt, sens), delai)
    return _classe_limite(r)


def _lim_txt(pt, sens):
    if pt == sp.oo:
        return r"x\to+\infty"
    if pt == -sp.oo:
        return r"x\to-\infty"
    return r"x\to " + sp.latex(pt) + (r"^{+}" if sens == "+" else r"^{-}")


def _fnum(f, x):
    """Fonction numérique (mpmath) de f, ou None."""
    try:
        return sp.lambdify(x, f, modules=["mpmath"])
    except Exception:
        return None


def _eval_mp(fm, v):
    try:
        r = fm(mp.mpf(v))
        if isinstance(r, mp.mpc):
            if abs(r.imag) > 1e-12:
                return None
            r = r.real
        return r
    except Exception:
        return None


def _quad(fm, a, b):
    """Intégrale numérique mpmath sur [a, b] (bornes éventuellement infinies), ou None."""
    try:
        mp.mp.dps = 20
        A = -mp.inf if a == -sp.oo else mp.mpf(float(sp.N(a)))
        B = mp.inf if b == sp.oo else mp.mpf(float(sp.N(b)))
        r = _avec_delai(lambda: mp.quad(lambda t: fm(t), [A, B]), 20)
        if r is None:
            return None
        if isinstance(r, mp.mpc):
            if abs(r.imag) > 1e-8:
                return None
            r = r.real
        r = float(r)
        # bornes infinies : contrôle par découpage (intégrandes oscillants)
        if A == -mp.inf or B == mp.inf:
            try:
                if B == mp.inf and A != -mp.inf:
                    pts = [A + k for k in range(0, 1001)]
                    X = pts[-1]
                    tete = _avec_delai(lambda: mp.quad(lambda t: fm(t), pts), 25)
                    queue = _avec_delai(lambda: mp.quad(lambda t: fm(t), [X, mp.inf]), 10)
                elif A == -mp.inf and B != mp.inf:
                    pts = [B - k for k in range(1000, -1, -1)]
                    X = pts[0]
                    tete = _avec_delai(lambda: mp.quad(lambda t: fm(t), pts), 25)
                    queue = _avec_delai(lambda: mp.quad(lambda t: fm(t), [-mp.inf, X]), 10)
                else:
                    tete = queue = None
                if tete is not None:
                    v2 = float(mp.re(tete))
                    q2 = float(mp.re(queue)) if queue is not None else 0.0
                    if abs(r - (v2 + q2)) > 1e-7 * (1 + abs(r)):
                        r = v2 if abs(q2) > 1e-3 * (1 + abs(v2)) or abs(r - v2) < abs(r - v2 - q2) else v2 + q2
                        r = v2
            except Exception:
                pass
        return r
    except Exception:
        return None


def _T(s):
    """'texte $maths$ texte' -> LaTeX (les segments de texte passent en \\text{})."""
    out = []
    for i, seg in enumerate(s.split("$")):
        if i % 2 == 0:
            if seg:
                seg = seg.replace("\\", "").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#").replace("_", r"\_")
                out.append(r"\text{" + seg + "}")
        elif seg:
            out.append(seg)
    return " ".join(out)




# -------------------------------------------------------------
# Analyse : points à problème, découpage en morceaux
# -------------------------------------------------------------
def _decoupage(f, x, a, b):
    """Renvoie (morceaux, infos) ou (None, message).
    morceaux : liste de dict(lo, hi, pb_lo, pb_hi) ; infos : liste de lignes de diagnostic."""
    I = sp.Interval(a, b)
    cd = _avec_delai(lambda: continuous_domain(f, x, I), 20)
    comp = None
    if cd is not None:
        comp = _avec_delai(lambda: I - cd, 10)
    if comp is None:
        # repli : points de raccord des Piecewise
        cands = set()
        try:
            for pw in f.atoms(sp.Piecewise):
                for e_, ens in pw.as_expr_set_pairs():
                    for bb in (getattr(ens, "start", None), getattr(ens, "end", None)):
                        if bb is not None and bb.is_finite and (_num(bb) is not None) and _num(a) <= _num(bb) <= _num(b):
                            cands.add(bb)
            for d_ in f.atoms(sp.Abs):
                sol = _avec_delai(lambda: sp.solveset(d_.args[0], x, domain=sp.Reals), 8)
                if isinstance(sol, sp.FiniteSet):
                    for bb in sol.args:
                        if _num(a) <= _num(bb) <= _num(b):
                            cands.add(bb)
        except Exception:
            pass
        if not cands:
            return None, "Le domaine de continuité de f sur l'intervalle n'a pas pu être déterminé."
        comp = sp.FiniteSet(*cands)
    if comp == S.EmptySet:
        cand = []
    elif isinstance(comp, sp.FiniteSet):
        cand = sorted(list(comp.args), key=lambda q: _num(q))
    else:
        return None, "f n'est pas définie (ou pas continue) sur tout un sous-intervalle de l'intervalle choisi : " + str(comp)
    infos = []
    interieurs = []
    pb_a = False
    pb_b = False
    diag = {}
    for q in cand:
        gauche = _lim(f, x, q, "-") if q != a else None
        droite = _lim(f, x, q, "+") if q != b else None
        ok = all((c is None) or (c[0] == "fin") for c in (gauche, droite))
        diag[q] = (gauche, droite, ok)
        if q == a:
            pb_a = not ok
        elif q == b:
            pb_b = not ok
        elif not ok:
            interieurs.append(q)
    pts = [a] + interieurs + [b]
    morceaux = []
    for i in range(len(pts) - 1):
        lo, hi = pts[i], pts[i + 1]
        pl = (lo in (sp.oo, -sp.oo)) or (i == 0 and pb_a) or (i > 0)
        ph = (hi in (sp.oo, -sp.oo)) or (i == len(pts) - 2 and pb_b) or (i < len(pts) - 2)
        morceaux.append(dict(lo=lo, hi=hi, pb_lo=bool(pl), pb_hi=bool(ph)))
    return morceaux, dict(cand=cand, diag=diag, interieurs=interieurs)


def _point_coupure(lo, hi):
    """Point c strictement intérieur, utilisé pour couper une intégrale à deux bornes impropres."""
    if lo == -sp.oo and hi == sp.oo:
        return sp.Integer(0)
    if lo == -sp.oo:
        return sp.simplify(hi - 1)
    if hi == sp.oo:
        return sp.simplify(lo + 1)
    return sp.simplify((lo + hi) / 2)


# -------------------------------------------------------------
# Primitive (validée numériquement)
# -------------------------------------------------------------
def _echantillons_dans(lo, hi):
    """Quatre points strictement à l'intérieur de ]lo, hi[."""
    if lo == -sp.oo and hi == sp.oo:
        return [-1.7, -0.6, 0.45, 1.3]
    if lo == -sp.oo:
        h = float(sp.N(hi))
        return [h - 4.1, h - 2.3, h - 1.1, h - 0.37]
    if hi == sp.oo:
        l = float(sp.N(lo))
        return [l + 0.37, l + 1.1, l + 2.3, l + 4.1]
    l, h = float(sp.N(lo)), float(sp.N(hi))
    return [l + (h - l) * t for t in (0.21, 0.37, 0.62, 0.83)]


def _valide_primitive(F, f, x, lo, hi):
    try:
        fm, Fm = _fnum(f, x), _fnum(F, x)
        if fm is None or Fm is None:
            return False
        pts = _echantillons_dans(lo, hi)
        mp.mp.dps = 20
        for i in range(len(pts) - 1):
            s1, s2 = mp.mpf(pts[i]), mp.mpf(pts[i + 1])
            exact = mp.quad(fm, [s1, s2])
            v1, v2 = Fm(s1), Fm(s2)
            for vv in (v1, v2):
                if isinstance(vv, mp.mpc) and abs(vv.imag) > 1e-9:
                    return False
            diff = (v2 - v1)
            if isinstance(diff, mp.mpc):
                diff = diff.real
            if abs(exact - diff) > 1e-6 * (1 + abs(exact)):
                return False
        return True
    except Exception:
        return False


def _primitive(f, x, lo, hi):
    """Renvoie F (primitive de f valable sur ]lo, hi[) ou None."""
    t = sp.Symbol("t", real=True)

    def directe():
        F = _avec_delai(lambda: sp.integrate(f, x), 25)
        if F is not None and not F.has(sp.Integral, sp.meijerg, sp.hyper, *_SPECF):
            F = _logabs(F)
            Fs = _avec_delai(lambda: sp.simplify(F), 10) or F
            if _valide_primitive(Fs, f, x, lo, hi):
                return Fs
            if _valide_primitive(F, f, x, lo, hi):
                return F
        return None

    def miroir():
        G = _avec_delai(lambda: sp.integrate(f.subs(x, -t), t), 25)
        if G is not None and not G.has(sp.Integral, sp.meijerg, sp.hyper, *_SPECF):
            F2 = _logabs(-G.subs(t, -x))
            if _valide_primitive(F2, f, x, lo, hi):
                return sp.simplify(F2)
        return None

    def branche(F):
        if F is None or not F.has(sp.Piecewise):
            return F
        try:
            pts = _echantillons_dans(lo, hi)
            s = sp.Float(pts[1])
            for e_, cond in F.args if isinstance(F, sp.Piecewise) else []:
                if cond == True or cond.subs(x, s) == True:
                    e2 = sp.simplify(e_)
                    return e2 if _valide_primitive(e2, f, x, lo, hi) else None
        except Exception:
            pass
        return None

    negatif = (hi != sp.oo) and (_num(hi) is not None) and _num(hi) <= 0
    ordre = (miroir, directe) if negatif else (directe, miroir)
    for essai in ordre:
        r = essai()
        if r is not None:
            return branche(r)
    return None


def _methode_texte(f, x):
    """Courte indication de méthode (ou chaîne vide)."""
    try:
        if f.is_rational_function(x):
            num, den = sp.fraction(sp.together(f))
            if sp.Poly(den, x).degree() >= 2:
                ap = sp.apart(f, x)
                if ap != f:
                    return r"\text{Décomposition en éléments simples : } f(x)=" + sp.latex(ap)
            return ""
        trig = [t for t in f.atoms(sp.sin, sp.cos, sp.exp, sp.log) if t.has(x)]
        if trig:
            tmp = f.xreplace({t: 1 for t in trig})
            if tmp.is_polynomial(x) and tmp.has(x):
                return r"\text{Le facteur } " + sp.latex(sp.expand(tmp)) + r" \text{ est un polynôme : intégration par parties (on dérive le polynôme).}"
        return ""
    except Exception:
        return ""


# -------------------------------------------------------------
# Signe, équivalents, nature de ∫|f|
# -------------------------------------------------------------
def _points_vers(c, p):
    """Points rapprochés de p (bord impropre) en partant de c, par fenêtres successives."""
    pts = []
    cf = float(sp.N(c)) if c not in (sp.oo, -sp.oo) else 0.0
    if p in (sp.oo, -sp.oo):
        sg = 1.0 if p == sp.oo else -1.0
        for k in range(1, 6):
            base = 10.0 ** k
            pts += [sg * base * (1 + 0.137 * j) for j in range(0, 25)]
    else:
        pf = float(sp.N(p))
        for k in range(1, 6):
            d = (cf - pf) * (10.0 ** (-k))
            pts += [pf + d * (1 + 0.137 * j) for j in range(0, 25)]
    return pts


def _signe_voisinage(fm, c, p):
    """+1, -1 (signe constant), 0 (changements de signe), None (indéterminé) ; avec deux valeurs témoins."""
    vals = []
    for t in _points_vers(c, p):
        v = _eval_mp(fm, t)
        if v is None or v == 0:
            continue
        vals.append((t, v))
    if len(vals) < 5:
        return None, None
    pos = sum(1 for _, v in vals if v > 0)
    neg = len(vals) - pos
    # deux témoins « ronds »
    if p in (sp.oo, -sp.oo):
        sg = 1.0 if p == sp.oo else -1.0
        tt = [sg * 10.0, sg * 1e5]
    else:
        cf, pf = float(sp.N(c)), float(sp.N(p))
        tt = [pf + (cf - pf) * 0.1, pf + (cf - pf) * 1e-5]
    tem = []
    for t_ in tt:
        v_ = _eval_mp(fm, t_)
        tem.append((t_, v_ if v_ is not None else vals[0][1]))
    if neg == 0:
        return 1, tuple(tem)
    if pos == 0:
        return -1, tuple(tem)
    return 0, tuple(tem)


def _equivalent(fpos, x, c, p):
    """f > 0 près de p. Renvoie dict(exposant, C, gamma) tel que f ~ C * u**e * (ln u)**gamma avec
    u = x (p=+oo), -x (p=-oo), 1/|x-p| (p fini)."""
    u = sp.Symbol("u", positive=True)
    if p == sp.oo:
        g = fpos.subs(x, u)
    elif p == -sp.oo:
        g = fpos.subs(x, -u)
    else:
        sgn = 1 if _num(c) > _num(p) else -1
        g = fpos.subs(x, p + sgn / u)
    try:
        g = sp.simplify(g)
    except Exception:
        pass
    r = _avec_delai(lambda: sp.limit(sp.log(g) / sp.log(u), u, sp.oo), 15)
    cl, e = _classe_limite(r)
    if cl != "fin":
        return None
    e = sp.nsimplify(e) if not e.is_Rational else e
    C = _avec_delai(lambda: sp.limit(sp.simplify(g / u ** e), u, sp.oo), 15)
    clC, Cv = _classe_limite(C)
    gam = sp.Integer(0)
    if clC == "fin" and Cv != 0:
        return dict(e=e, C=sp.simplify(Cv), gamma=gam, g=g, u=u)
    # facteur logarithmique éventuel
    gl = _avec_delai(lambda: sp.limit(sp.log(sp.simplify(g / u ** e)) / sp.log(sp.log(u)), u, sp.oo), 15)
    clg, gv = _classe_limite(gl)
    if clg == "fin":
        gv = sp.nsimplify(gv)
        C2 = _avec_delai(lambda: sp.limit(sp.simplify(g / (u ** e * sp.log(u) ** gv)), u, sp.oo), 15)
        cl2, C2v = _classe_limite(C2)
        if cl2 == "fin" and C2v != 0:
            return dict(e=e, C=sp.simplify(C2v), gamma=gv, g=g, u=u)
    return None


def _verdict_equivalent(eq, p):
    """True si l'intégrale (positive) de f près de p converge, d'après l'équivalent."""
    m = 0 if p in (sp.oo, -sp.oo) else -2
    s = eq["e"] + m
    if s < -1:
        return True
    if s > -1:
        return False
    return bool(eq["gamma"] < -1)


def _texte_equivalent(eq, x, p, signe):
    """Ligne LaTeX : f(x) ~ C / |x-p|^alpha, et critère de comparaison appliqué."""
    e, C, gam = eq["e"], eq["C"], eq["gamma"]
    pre = (r"-" if signe < 0 else "")
    coef = "" if C == 1 else sp.latex(C) + r"\cdot "

    def puiss(var, al):
        return var if al == 1 else var + r"^{" + sp.latex(al) + r"}"

    def lnp(base, g):
        return base if g == 1 else base + r"^{" + sp.latex(g) + r"}"

    if p in (sp.oo, -sp.oo):
        alpha = sp.simplify(-e)
        var = r"x" if p == sp.oo else r"|x|"
        if gam > 0:
            num_, den = lnp(r"(\ln " + var + r")", gam), puiss(var, alpha)
        else:
            num_, den = "1", puiss(var, alpha) + (r"\,(\ln " + var + r")^{" + sp.latex(-gam) + r"}" if gam != 0 else "")
        l1 = r"f(x)\ \underset{" + _lim_txt(p, None) + r"}{\sim}\ " + pre + coef + (num_ if (den == "x^{0}" or den == "|x|^{0}" or alpha == 0) else r"\frac{" + num_ + "}{" + den + r"}")
        if gam != 0 and alpha == 1:
            crit = r"\text{Bertrand : } \int^{\infty}\frac{dx}{x(\ln x)^{\beta}} \text{ converge} \iff \beta>1 \text{ ; ici } \beta=" + sp.latex(-gam)
        else:
            crit = (r"\text{Riemann en } " + (r"+\infty" if p == sp.oo else r"-\infty") + r"\text{ : } \int^{\infty}\frac{dx}{|x|^{\alpha}} \text{ converge} \iff \alpha>1 \text{ ; ici } \alpha="
                    + sp.latex(alpha))
        return l1, crit
    alpha = e
    d = sp.latex(sp.Abs(sp.simplify(x - p)))
    if gam > 0:
        num_, den = lnp(r"|\ln " + d + r"|", gam), puiss(d, alpha)
    else:
        num_, den = "1", puiss(d, alpha) + (r"\,|\ln " + d + r"|^{" + sp.latex(-gam) + r"}" if gam != 0 else "")
    l1 = r"f(x)\ \underset{x\to " + sp.latex(p) + r"}{\sim}\ " + pre + coef + (num_ if (den == "x^{0}" or den == "|x|^{0}" or alpha == 0) else r"\frac{" + num_ + "}{" + den + r"}")
    if gam != 0 and alpha == 1:
        crit = r"\text{Bertrand : } \int\frac{dx}{" + d + r"\,|\ln " + d + r"|^{\beta}} \text{ converge} \iff \beta>1 \text{ ; ici } \beta=" + sp.latex(-gam)
    else:
        crit = (r"\text{Riemann en } " + sp.latex(p) + r"\text{ : } \int\frac{dx}{" + d + r"^{\alpha}} \text{ converge} \iff \alpha<1 \text{ ; ici } \alpha=" + sp.latex(alpha))
    return l1, crit


def _domination(fpos_abs, x, c, p):
    """|f| = o(1/x^2) en l'infini, ou o(1/|x-p|^(1/2)) en p fini ? Renvoie (vrai, genre) ou (False, None)."""
    u = sp.Symbol("u", positive=True)
    if p == sp.oo:
        g, s = fpos_abs.subs(x, u), -2
    elif p == -sp.oo:
        g, s = fpos_abs.subs(x, -u), -2
    else:
        sgn = 1 if _num(c) > _num(p) else -1
        g, s = fpos_abs.subs(x, p + sgn / u), sp.Rational(1, 2)
    r = _avec_delai(lambda: sp.limit(sp.simplify(g / u ** s), u, sp.oo), 15)
    cl, v = _classe_limite(r)
    if cl == "fin" and v == 0:
        return True, ("inf" if p in (sp.oo, -sp.oo) else "fin")
    return False, None




# -------------------------------------------------------------
# Rédaction du calcul d'une primitive (parties, changement de variable, éléments simples ...)
# -------------------------------------------------------------
_CPT = {}


def _var(cle, *options):
    """Choisit tour à tour parmi plusieurs formulations (évite que deux phrases voisines se ressemblent)."""
    i = _CPT.get(cle, 0)
    _CPT[cle] = i + 1
    return options[i % len(options)]


def _aff(e):
    """LaTeX d'une primitive : les logarithmes s'écrivent ln|.| (affichage des primitives de 1/x)."""
    try:
        e2 = e.replace(lambda t: isinstance(t, sp.log) and not isinstance(t.args[0], sp.Abs) and t.args[0].is_positive is not True,
                       lambda t: sp.log(sp.Abs(t.args[0])))
        return sp.latex(e2)
    except Exception:
        return sp.latex(e)


def _usuelle(g, var):
    """Primitive de g (calcul rapide), ou None."""
    r = _avec_delai(lambda: sp.integrate(g, var), 10)
    if r is None or r.has(sp.Integral, sp.meijerg, sp.hyper):
        return None
    try:
        return sp.simplify(r)
    except Exception:
        return r


def _narr_puissance(f, x):
    """f polynôme ou somme de puissances : terme à terme."""
    sv = sp.latex(x)
    try:
        fe = sp.expand(f)
        termes = sp.Add.make_args(fe)
        if not all(t.is_polynomial(x) or (sp.simplify(t * x ** 5).is_polynomial(x) and not t.has(sp.sin, sp.cos, sp.exp, sp.log)) for t in termes):
            return None
        if len(termes) == 1 and (fe == x or fe.is_Number):
            return None
        F = sp.integrate(fe, x)
        if len(termes) == 1:
            t = termes[0]
            if not (t.is_Pow and t.base == x) and not (t.is_Mul and any(a.is_Pow and a.base == x for a in t.args)) and t != x:
                return None
            return [_T("L'intégrande est une puissance de $" + sv + "$ : on utilise $\\int " + sv + "^{\\alpha}\\,d" + sv + "=\\frac{" + sv + "^{\\alpha+1}}{\\alpha+1}$ ($\\alpha\\ne-1$), ce qui donne $F(" + sv + ")=" + sp.latex(F) + "$.")]
        return [_T("On intègre terme à terme, avec $\\int " + sv + "^{\\alpha}\\,d" + sv + "=\\frac{" + sv + "^{\\alpha+1}}{\\alpha+1}$ pour $\\alpha\\ne-1$ : $F(" + sv + ")=" + sp.latex(F) + "$.")]
    except Exception:
        return None


def _narr_rationnel(f, x):
    """Fraction rationnelle : éléments simples puis primitives usuelles."""
    sv = sp.latex(x)
    try:
        if not f.is_rational_function(x):
            return None
        num, den = sp.fraction(sp.together(f))
        if sp.Poly(den, x).degree() < 2:
            return None
        ap = sp.apart(f, x)
        if ap == f or len(sp.Add.make_args(ap)) < 2:
            return None
        Ft = 0
        for t in sp.Add.make_args(ap):
            r = _usuelle(t, x)
            if r is None:
                return None
            Ft += r
        return [_T("On décompose en éléments simples : $" + sp.latex(f) + "=" + sp.latex(ap) + "$."),
                _T("Chaque terme admet une primitive usuelle ($\\int\\frac{d" + sv + "}{" + sv + "-a}=\\ln|" + sv + "-a|$, $\\int\\frac{d" + sv + "}{(" + sv + "-a)^{n}}=\\frac{-1}{(n-1)(" + sv + "-a)^{n-1}}$, $\\int\\frac{d" + sv + "}{" + sv + "^2+1}=\\arctan " + sv + "$), d'où $F(" + sv + ")=" + _aff(Ft) + "$.")]
    except Exception:
        return None


def _choisir(alts):
    """Parmi des alternatives de manualintegrate, la rédaction la plus naturelle."""
    def cle(r):
        n = type(r).__name__
        pref = 0 if n in ("PartsRule", "CyclicPartsRule") else 1
        poly = 1
        try:
            if n == "URule":
                poly = 0 if r.u_func.is_polynomial(r.variable) else 1
        except Exception:
            poly = 1
        return (pref, poly, len(repr(r)))
    return sorted(alts, key=cle)[0]


_F0 = [None]


def _nr(rule, var, prof=0):
    """Lignes rédigées pour une règle de manualintegrate (variable var). Renvoie (lignes, primitive)."""
    nom = type(rule).__name__
    v0 = getattr(rule, "variable", None)

    def sb(e):
        try:
            return e.subs(v0, var) if v0 is not None else e
        except Exception:
            return e
    if prof > 4:
        return [], None
    if nom == "AlternativeRule":
        return _nr(_choisir(rule.alternatives), var, prof)
    integrand = sb(rule.integrand) if getattr(rule, "integrand", None) is not None else None
    if nom == "URule":
        uf = sb(rule.u_func)
        w = sp.Symbol("t")
        sous_integrand = rule.substep.integrand.subs(rule.u_var, w)
        sous = rule.substep
        disp = _F0[0] if (prof == 0 and _F0[0] is not None) else integrand
        lg = [_T("On pose $t=" + sp.latex(uf) + "$, donc $dt=" + sp.latex(sp.simplify(sp.diff(uf, var))) + "\\,d" + sp.latex(var) + "$ : $\\int " + sp.latex(disp) + "\\,d" + sp.latex(var)
                 + "=\\int " + sp.latex(sp.simplify(sous_integrand)) + "\\,dt$.")]
        lg2, G = _nr_sous(sous, w, prof + 1)
        lg += lg2
        if G is None:
            G = _usuelle(sous_integrand, w)
        if G is None:
            return lg, None
        F = G.subs(w, uf)
        return lg, F
    if nom == "PartsRule":
        u = sb(rule.u)
        dv = sb(rule.dv)
        du = sp.simplify(sp.diff(u, var))
        vv = _usuelle(dv, var)
        if vv is None:
            return [], None
        reste = sp.simplify(du * vv)
        neg = sp.simplify(-reste)
        nu, nv = (("p", "q") if getattr(var, "name", "") in ("u", "v", "p", "q") else ("u", "v"))
        lg = [_T("Intégration par parties : on pose $" + nu + "=" + sp.latex(u) + "$ et $" + nv + "'=" + sp.latex(dv) + "$, d'où $" + nu + "'=" + sp.latex(du) + "$ et $" + nv + "=" + sp.latex(vv) + "$. Ainsi $\\int " + sp.latex(integrand)
                 + "\\,d" + sp.latex(var) + "=" + sp.latex(sp.simplify(u * vv)) + (("-\\int " + sp.latex(sp.simplify(-neg))) if neg.could_extract_minus_sign() else ("+\\int " + sp.latex(neg))) + "\\,d" + sp.latex(var) + "$.")]
        sec = rule.second_step
        Gn = None
        if sec is not None and type(sec).__name__ == "PartsRule" and prof < 3:
            try:
                from sympy.integrals.manualintegrate import integral_steps
                sous = _avec_delai(lambda: integral_steps(neg, var), 10)
                if sous is not None and type(sous).__name__ in ("PartsRule", "AlternativeRule", "ConstantTimesRule"):
                    lg.append(_T("Il reste $\\int " + sp.latex(sp.simplify(-neg) if neg.could_extract_minus_sign() else neg) + "\\,d" + sp.latex(var) + "$ : une seconde intégration par parties l'abaisse d'un degré."))
                    lg2, Gn = _nr(sous, var, prof + 1)
                    lg += lg2
            except Exception:
                Gn = None
        if Gn is None:
            Gn = _usuelle(neg, var)
        if Gn is None:
            return lg, None
        F = sp.simplify(u * vv + Gn)
        return lg, F
    if nom == "CyclicPartsRule":
        F = _usuelle(integrand, var)
        lg = [_T("Deux intégrations par parties successives font réapparaître l'intégrale cherchée $I=\\int " + sp.latex(integrand) + "\\,d" + sp.latex(var) + "$ dans le second membre : on résout l'équation en $I$.")]
        return lg, F
    if nom == "ConstantTimesRule":
        c = rule.constant
        autre = sb(rule.other)
        lg = [_T("On factorise la constante : $\\int " + sp.latex(integrand) + "\\,d" + sp.latex(var) + "=" + sp.latex(c) + "\\int " + sp.latex(autre) + "\\,d" + sp.latex(var) + "$.")]
        lg2, G = _nr(rule.substep, var, prof + 1)
        lg += lg2
        if G is None:
            G = _usuelle(autre, var)
        if not lg2 and G is not None:
            lg.append(_T("Or $\\int " + sp.latex(autre) + "\\,d" + sp.latex(var) + "=" + _aff(G) + "$ (primitive usuelle)."))
        return lg, (None if G is None else sp.simplify(c * G))
    if nom == "RewriteRule":
        try:
            a_ = sp.simplify(integrand)
            b_ = sb(rule.rewritten)
            trivial = (a_ == b_) or (sp.simplify(a_ - b_) == 0 and a_.is_Pow and b_.is_Pow)
        except Exception:
            trivial = False
        lg = [] if trivial else [_T("On réécrit $" + sp.latex(sp.simplify(integrand)) + "=" + sp.latex(sb(rule.rewritten)) + "$.")]
        lg2, G = _nr(rule.substep, var, prof + 1)
        return lg + lg2, G
    if nom == "AddRule":
        F = _usuelle(integrand, var)
        return [_T("On intègre chaque terme de la somme séparément.")], F
    # primitives usuelles (règle terminale)
    F = _usuelle(integrand, var) if integrand is not None else None
    return [], F


def _nr_sous(rule, var, prof):
    """Rédaction du sous-problème après changement de variable (on ne répète pas les primitives usuelles)."""
    nom = type(rule).__name__
    if nom in ("AlternativeRule",):
        rule = _choisir(rule.alternatives)
        nom = type(rule).__name__
    if nom in ("PowerRule", "ExpRule", "ReciprocalRule", "ArctanRule", "ArcsinRule", "TrigRule", "ConstantRule", "HyperbolicRule"):
        g = rule.integrand.subs(rule.variable, var) if getattr(rule, "variable", None) is not None else rule.integrand
        G = _usuelle(g, var)
        if G is None:
            return [], None
        return [_T("C'est une primitive usuelle : $\\int " + sp.latex(g) + "\\,dt=" + _aff(G) + "$.")], G
    lg, G = _nr(rule, var, prof)
    return lg, G


def _narrer_primitive(f, x, F):
    """Rédaction de la recherche d'une primitive de f ; [] si aucune rédaction fiable."""
    try:
        lignes = _narr_puissance(f, x) or _narr_rationnel(f, x)
        if lignes:
            return lignes
        from sympy.integrals.manualintegrate import integral_steps
        regle = _avec_delai(lambda: integral_steps(f, x), 12)
        if regle is None:
            return []
        _F0[0] = f
        lg, Fm = _nr(regle, x)
        _F0[0] = None
        if Fm is None:
            return []
        if not lg:
            lg = [_T("C'est une primitive usuelle : $\\int " + sp.latex(f) + "\\,d" + sp.latex(x) + "=" + _aff(Fm) + "$.")]
        # cohérence avec la primitive retenue (écart constant)
        d = sp.simplify(sp.diff(Fm - F, x))
        if d != 0:
            ok = False
            try:
                autres = {s_: 3 for s_ in d.free_symbols if s_ != x}
                d = d.subs(autres)
                ok = all(abs(complex(sp.N(d.subs(x, t)))) < 1e-9 for t in (0.37, 1.31, 2.7))
            except Exception:
                ok = False
            if not ok:
                return []
        lg.append(_T("On en déduit une primitive : $F(" + sp.latex(x) + ")=" + _aff(F) + "$."))
        return lg[:9]
    except Exception:
        return []


# -------------------------------------------------------------
# Chaîne d'égalités alignées pour le calcul d'une intégrale d'une variable
# (parties, changement de variable, éléments simples ... avec justification brève à droite)
# -------------------------------------------------------------
_RACINE_Q = [None]


def _bloc_chaine(debut, lignes):
    """debut : membre de gauche ; lignes : [(expression, commentaire)] ; renvoie un bloc LaTeX aligné."""
    corps = []
    for i, (expr, com) in enumerate(lignes):
        s = (debut if i == 0 else "") + " &= " + _fin_coef(expr)
        if com:
            s += r" && " + _T("(" + com + ")")
        corps.append(s)
    return r"\begin{aligned} " + r" \\ ".join(corps) + r" \end{aligned}"


def _logabs(e):
    """Remplace log(t) par log(|t|) quand t n'est pas connu positif (primitive réelle de 1/t)."""
    try:
        return e.replace(lambda t: isinstance(t, sp.log) and not isinstance(t.args[0], sp.Abs) and t.args[0].is_positive is not True,
                         lambda t: sp.log(sp.Abs(t.args[0])))
    except Exception:
        return e


import re as _re_q
_MARQ = _re_q.compile("^\u27e6([^\u27e7]*)\u27e7")


def _additif(L):
    """Vrai si L contient un + ou un - au premier niveau (hors crochets, parenthèses \\left...\\right et accolades)."""
    prof = 0
    i = 0
    n = len(L)
    debut = True
    while i < n:
        if L.startswith("\\left", i):
            prof += 1
            i += 5
            continue
        if L.startswith("\\right", i):
            prof -= 1
            i += 6
            continue
        c = L[i]
        if c == "{":
            prof += 1
        elif c == "}":
            prof -= 1
        elif prof == 0 and c in "+-" and not debut:
            # opérateur binaire s'il est précédé d'un espace
            if i > 0 and L[i - 1] == " ":
                return True
        if c != " ":
            debut = False
        i += 1
    return False


def _mulc(k, L):
    """Coefficient k devant L ; les coefficients successifs sont fusionnés (marqueurs remplacés à l'affichage)."""
    m = _MARQ.match(L)
    rest = L
    kk = k
    if m:
        try:
            kk = sp.simplify(k * sp.sympify(m.group(1)))
        except Exception:
            kk = k
        rest = L[m.end():]
    if kk == 1:
        return rest
    if _additif(rest):
        rest = r"\left(" + rest + r"\right)"
    return "\u27e6" + sp.srepr(kk) + "\u27e7" + rest


def _joint(base, sgn, L):
    L = L.strip()
    m = _MARQ.match(L)
    if m:
        try:
            kk = sp.sympify(m.group(1))
            if kk.could_extract_minus_sign():
                sgn = "+" if sgn == "-" else "-"
                L = (_mulc(-kk, L[m.end():]) if -kk != 1 else L[m.end():])
        except Exception:
            pass
    elif L.startswith("-"):
        sgn = "+" if sgn == "-" else "-"
        L = L[1:].strip()
    return base + " " + sgn + " " + L


def _fin_coef(s):
    def rem(m):
        try:
            kk = sp.sympify(m.group(1))
        except Exception:
            return ""
        if kk == 1:
            return ""
        if kk == -1:
            return "-"
        sk = sp.latex(kk)
        if getattr(kk, "is_Add", False):
            sk = r"\left(" + sk + r"\right)"
        return sk + r"\,"
    return _re_q.sub("\u27e6([^\u27e7]*)\u27e7", rem, s)


def _crochet(P, A, B):
    return r"\left[" + _aff(P) + r"\right]_{" + _borne_lat(A) + "}^{" + _borne_lat(B) + "}"


def _iq(g, v, A, B):
    sg = sp.latex(g)
    if getattr(g, "is_Add", False):
        sg = r"\left(" + sg + r"\right)"
    return r"\int_{" + _borne_lat(A) + "}^{" + _borne_lat(B) + "}" + sg + r"\,d" + sp.latex(v)


def _nouveau_symbole(noms, exclus):
    for n_ in noms:
        if all(getattr(s_, "name", None) != n_ for s_ in exclus):
            return sp.Symbol(n_, real=True)
    return sp.Symbol(noms[0] + "_0", real=True)


class _Abandon(Exception):
    pass


def _croissant_sur(u_func, var, A, B, ctx):
    """u_func est de classe C1 et de dérivée de signe constant non nulle sur l'intervalle (échantillons)."""
    try:
        d = sp.diff(u_func, var)
        autres = {q: (sp.Rational(7, 3) if q.is_positive is not False else sp.Rational(-7, 3)) for q in d.free_symbols if q != var}
        if autres:
            d = d.subs(autres)
        a_, b_ = A, B
        cib = ctx.get("cible")
        sy = ctx.get("X")
        if sy is not None and cib is not None:
            a_ = a_.subs(sy, cib) if hasattr(a_, "subs") else a_
            b_ = b_.subs(sy, cib) if hasattr(b_, "subs") else b_
        na, nb = _num(a_), _num(b_)
        if a_ == -sp.oo:
            na = -sp.oo
        if b_ == sp.oo:
            nb = sp.oo
        if na is None or nb is None:
            return False
        lo_, hi_ = (a_, b_) if na <= nb else (b_, a_)
        pts = _echantillons_dans(lo_, hi_)
        sg = set()
        for p_ in pts:
            val = complex(sp.N(d.subs(var, p_)))
            if abs(val.imag) > 1e-12 or abs(val.real) < 1e-12 or val != val:
                return False
            sg.add(val.real > 0)
        return len(sg) == 1
    except Exception:
        return False


def _affine(e, var):
    try:
        return sp.Poly(e, var).degree() <= 1
    except Exception:
        return False


_INTERDITS = ("DontKnowRule", "DerivativeRule", "HeavisideRule", "DiracDeltaRule", "OrthogonalPolyRule", "JacobiRule", "GegenbauerRule",
              "ChebyshevTRule", "ChebyshevURule", "LegendreRule", "HermiteRule", "LaguerreRule", "AssocLaguerreRule", "IRule", "CiRule",
              "ChiRule", "EiRule", "SiRule", "ShiRule", "LiRule", "ErfRule", "FresnelCRule", "FresnelSRule", "PolylogRule", "UpperGammaRule",
              "EllipticFRule", "EllipticERule")


def _aplatir(rule):
    if type(rule).__name__ == "AlternativeRule":
        out = []
        for r_ in rule.alternatives:
            out += _aplatir(r_)
        return out
    return [rule]


def _cle_alt(r):
    n = type(r).__name__
    poly = 0
    try:
        if n == "URule":
            poly = 0 if r.u_func.is_polynomial(r.variable) else 1
    except Exception:
        poly = 1
    return (0 if n in ("PartsRule", "CyclicPartsRule", "RewriteRule") else 1, poly, len(repr(r)))


def _score_lignes(rows):
    sc = len(rows)
    for _l, com in rows:
        if "changement" in com or "$t=" in com:
            sc += 1.5
            if "e^{" in com.split(",")[0] or "\\sqrt" in com.split(",")[0] or "\\frac" in com.split(",")[0]:
                sc += 0.4
    return sc


class _TabRule:
    """IPP répétée d'un polynôme par une exponentielle / un sinus / un cosinus de variable affine."""
    def __init__(self, g, var, P, E, d):
        self.integrand, self.variable, self.P, self.E, self.d = g, var, P, E, d

    def __repr__(self):
        return "_TabRule(%s)" % self.integrand


def _alternatives_synthetiques(g, var):
    """Règles supplémentaires : éléments simples, parties directes pour log / arctan / arcsin."""
    out = []
    try:
        from sympy.integrals import manualintegrate as M
        if g.as_independent(var, as_Add=False)[0] != 1:
            return out
        if g.is_rational_function(var):
            num, den = sp.fraction(sp.together(g))
            if sp.Poly(den, var).degree() >= 2:
                ap = sp.apart(g, var)
                termes = sp.Add.make_args(ap)
                if len(termes) >= 2:
                    subs = []
                    for t_ in termes:
                        rr = _avec_delai(lambda t_=t_: M.integral_steps(t_, var), 8)
                        if rr is None:
                            subs = None
                            break
                        subs.append(rr)
                    if subs:
                        out.append(M.RewriteRule(integrand=g, variable=var, rewritten=ap, substep=M.AddRule(integrand=ap, variable=var, substeps=subs)))
        facteurs = sp.Mul.make_args(g)
        pol = [q for q in facteurs if q.has(var) and q.is_polynomial(var)]
        aut = [q for q in facteurs if not (q.has(var) and q.is_polynomial(var))]
        if pol and aut:
            P_ = sp.Mul(*pol)
            R_ = sp.Mul(*aut)
            ok_r = R_.has(var) and all(
                isinstance(q, (sp.exp, sp.sin, sp.cos, sp.sinh, sp.cosh)) and q.args[0].is_polynomial(var) and sp.Poly(q.args[0], var).degree() == 1
                or not q.has(var) for q in sp.Mul.make_args(R_))
            if ok_r and sp.Poly(sp.expand(P_), var).degree() >= 1:
                out.append(M.PartsRule(integrand=g, variable=var, u=P_, dv=R_, v_step=None, second_step=None))
                dP = sp.Poly(sp.expand(P_), var).degree()
                if dP >= 3 and len([q for q in sp.Mul.make_args(R_) if q.has(var)]) == 1:
                    out.append(_TabRule(g, var, sp.expand(P_), R_, dP))
        for i, fa in enumerate(facteurs):
            if (isinstance(fa, (sp.log, sp.atan, sp.asin, sp.acos)) and fa.args[0] == var):
                reste = sp.Mul(*[q for j, q in enumerate(facteurs) if j != i])
                if reste.is_polynomial(var) or (reste.is_Pow and reste.base == var and reste.exp != -1) or (reste.is_Mul and all(q.is_polynomial(var) or (q.is_Pow and q.base == var) for q in reste.args)):
                    if reste != -1 / var and sp.simplify(reste * var) != 1:
                        out.append(M.PartsRule(integrand=g, variable=var, u=fa, dv=reste, v_step=None, second_step=None))
    except Exception:
        pass
    return out


def _applique(rule, g, var, A, B, wrap, rows, ctx, prof=0, fus=False):
    try:
        k_, g2_ = g.as_independent(var, as_Add=False)
        if k_ != 1 and g2_.has(var) and "ConstantTimesRule" not in [type(r).__name__ for r in _aplatir(rule)]:
            from sympy.integrals import manualintegrate as _M
            sub_ = _avec_delai(lambda: _M.integral_steps(g2_, var), 10)
            if sub_ is not None:
                rule = _M.ConstantTimesRule(integrand=g, variable=var, constant=k_, other=g2_, substep=sub_)
    except Exception:
        pass
    alts = _aplatir(rule) + _alternatives_synthetiques(g, var)
    if len(alts) == 1:
        return _applique_un(alts[0], g, var, A, B, wrap, rows, ctx, prof, fus)
    meilleur = None
    for r_ in sorted(alts, key=lambda r: _cle_alt(r)):
        rows2 = []
        ctx2 = dict(ctx)
        try:
            G_ = _applique_un(r_, g, var, A, B, wrap, rows2, ctx2, prof, fus)
        except _Abandon:
            continue
        except Exception:
            continue
        sc = _score_lignes(rows2)
        if meilleur is None or sc < meilleur[0]:
            meilleur = (sc, rows2, ctx2, G_)
    if meilleur is None:
        raise _Abandon()
    rows.extend(meilleur[1])
    ctx.clear()
    ctx.update(meilleur[2])
    return meilleur[3]


def _detecte_cyclique(g, var, u1, dv1, v1, h1):
    """(u2, dv2, v2, rho) si deux IPP successives de g font réapparaître rho*g, sinon None."""
    try:
        if not (g.has(sp.sin, sp.cos, sp.sinh, sp.cosh) and g.has(sp.exp, sp.sin, sp.cos)):
            return None
        k, core = h1.as_independent(var, as_Add=False)
        for fa in sp.Mul.make_args(core):
            if not fa.has(var):
                continue
            if not isinstance(fa, (sp.sin, sp.cos, sp.sinh, sp.cosh)):
                continue
            u2 = sp.simplify(k * fa)
            dv2 = sp.simplify(h1 / u2)
            v2 = _usuelle_q(dv2, var)
            if v2 is None:
                continue
            h2 = sp.simplify(sp.diff(u2, var) * v2)
            ratio = sp.simplify(h2 / g)
            if (not ratio.has(var)) and ratio != 0 and sp.simplify(1 - ratio) != 0:
                return (u2, dv2, v2, ratio)
    except Exception:
        return None
    return None


def _rediger_cyclique(g, var, A, B, wrap, rows, ctx, u1, dv1, v1, cyc):
    u2, dv2, v2, rho = cyc
    h1 = sp.simplify(sp.diff(u1, var) * v1)
    P1 = sp.factor_terms(sp.expand(u1 * v1))
    P2 = sp.factor_terms(sp.expand(u1 * v1 - u2 * v2))
    neg1 = bool(h1.could_extract_minus_sign())
    base1 = _crochet(P1, A, B)
    rows.append((wrap(_joint(base1, "+" if neg1 else "-", _iq(sp.simplify(-h1) if neg1 else h1, var, A, B))),
                 "IPP : $u=" + sp.latex(u1) + "$, $v'=" + sp.latex(dv1) + "$"))
    base2 = _crochet(P2, A, B)
    cc = sp.simplify(rho)
    sgn2 = "-" if cc.could_extract_minus_sign() else "+"
    ac = sp.simplify(abs(cc)) if cc.is_number else sp.simplify(-cc if cc.could_extract_minus_sign() else cc)
    rows.append((wrap(_joint(base2, sgn2, _mulc(ac, _iq(g, var, A, B)) if ac != 1 else _iq(g, var, A, B))),
                 "seconde IPP : $u=" + sp.latex(u2) + "$, $v'=" + sp.latex(dv2) + "$ ; l'intégrale de départ réapparaît"))
    G = sp.factor_terms(sp.expand(P2 / (1 - cc)))
    k1 = sp.simplify(1 - cc)
    sk1 = sp.latex(k1) if not getattr(k1, "is_Add", False) else "\\left(" + sp.latex(k1) + "\\right)"
    rows.append((wrap(_crochet(G, A, B)), "on isole l'intégrale : $" + (sk1 if k1 != 1 else "") + "\\int=" + _aff(P2) + "$"))
    ctx["monde"] = (var, A, B, G)
    return G


def _applique_un(rule, g, var, A, B, wrap, rows, ctx, prof=0, fus=False):
    """Rédige l'intégrale de g sur [A,B] ; renvoie une primitive de g (variable var). rows reçoit (ligne, commentaire)."""
    if prof > 40:
        raise _Abandon()
    nom = type(rule).__name__
    v0 = getattr(rule, "variable", None)

    def sb(e):
        try:
            return e.subs(v0, var) if (v0 is not None and v0 != var) else e
        except Exception:
            return e

    sv = sp.latex(var)

    def terminal(G, com):
        if G is None:
            raise _Abandon()
        rows.append((wrap(_crochet(G, A, B)), com))
        ctx["monde"] = (var, A, B, G)
        return G

    if nom == "PiecewiseRule":
        for sub_, cond_ in rule.subfunctions:
            try:
                ok_ = (cond_ == True) or (sp.simplify(cond_) == True)
            except Exception:
                ok_ = False
            if ok_:
                return _applique(sub_, g, var, A, B, wrap, rows, ctx, prof + 1, fus)
        raise _Abandon()
    if nom == "_TabRule":
        P_, E_, d_ = rule.P, rule.E, rule.d
        V = [E_]
        for _i in range(d_ + 1):
            Vn = _usuelle_q(V[-1], var)
            if Vn is None:
                raise _Abandon()
            V.append(Vn)
        termes = []
        G = sp.Integer(0)
        for kk in range(d_ + 1):
            uk = sp.diff(P_, var, kk)
            if uk == 0:
                continue
            tk = sp.simplify(uk * V[kk + 1])
            G += (-1) ** kk * tk
            sg = "-" if kk % 2 else "+"
            def pl(e_):
                t_ = sp.latex(e_)
                try:
                    neg_ = e_.could_extract_minus_sign()
                except Exception:
                    neg_ = False
                return r"\left(" + t_ + r"\right)" if (getattr(e_, "is_Add", False) or neg_) else t_
            termes.append((sg, sp.latex(V[kk + 1]) if uk == 1 else pl(uk) + r"\," + pl(V[kk + 1])))
        corps = ""
        for i_, (sg, t_) in enumerate(termes):
            corps += (("- " if sg == "-" else "") if i_ == 0 else (" - " if sg == "-" else " + ")) + t_
        brut = r"\left[" + corps + r"\right]_{" + _borne_lat(A) + "}^{" + _borne_lat(B) + "}"
        rows.append((wrap(brut), "$" + str(d_) + "$ IPP successives : $u=" + sp.latex(P_) + "$ dérivé, $v'=" + sp.latex(E_) + "$ intégré à chaque étape"))
        Gs = sp.simplify(G)
        ap = wrap(_crochet(Gs, A, B))
        if ap != wrap(brut):
            rows.append((ap, "on regroupe"))
        ctx["monde"] = (var, A, B, Gs)
        return Gs
    if nom == "ConstantTimesRule":
        k = rule.constant
        autre = sb(rule.other)
        rows.append((wrap(_mulc(k, _iq(autre, var, A, B))), "on sort la constante $" + sp.latex(k) + "$"))
        Gs = _applique(rule.substep, autre, var, A, B, lambda L: wrap(_mulc(k, L)), rows, ctx, prof + 1, fus)
        Gk = k * Gs
        if not fus:
            ap = wrap(_crochet(Gk, A, B))
            if not rows or rows[-1][0] != ap:
                rows.append((ap, "on rentre la constante"))
            ctx["monde"] = (var, A, B, Gk)
        return Gk
    if nom == "AddRule":
        from sympy.integrals.manualintegrate import integral_steps as _is2
        termes = [sb(s_.integrand) for s_ in rule.substeps]
        items = [t_.as_independent(var, as_Add=False) for t_ in termes]
        n = len(items)
        def aff_t(k, t2):
            return _mulc(k, _iq(t2, var, A, B))
        def somme(morceaux):
            base = morceaux[0]
            for m_ in morceaux[1:]:
                base = _joint(base, "+", m_)
            return base
        com_add = ctx.pop("com_add", "linéarité")
        rows.append((wrap(somme([aff_t(k, t2) for k, t2 in items])), com_add))
        faits = []
        Gs = []
        for i, (k, t2) in enumerate(items):
            def wi(L, i=i, k=k, faits=faits):
                morceaux = list(faits) + [_mulc(k, L)] + [aff_t(kk, tt) for kk, tt in items[i + 1:]]
                return wrap(somme(morceaux))
            sub = _avec_delai(lambda t2=t2: _is2(t2, var), 10)
            if sub is None:
                raise _Abandon()
            Gi = _applique(sub, t2, var, A, B, wi, rows, ctx, prof + 1, True)
            Gs.append(k * Gi)
            faits.append(_crochet(k * Gi, A, B))
        G = sum(Gs, sp.Integer(0))
        if n > 1 and not fus:
            rows.append((wrap(_crochet(G, A, B)), "on regroupe"))
        ctx["monde"] = (var, A, B, G)
        return G
    if nom == "RewriteRule":
        new = sb(rule.rewritten)
        try:
            a_ = sp.simplify(g)
            trivial = (g == new) or (a_ == new) or (sp.simplify(a_ - new) == 0 and a_.is_Pow and new.is_Pow)
        except Exception:
            trivial = False
        try:
            if not trivial and not (new.is_Add and not g.is_Add) and ({type(q).__name__ for q in new.atoms(sp.Function)} == {type(q).__name__ for q in g.atoms(sp.Function)}):
                trivial = True
        except Exception:
            pass
        if not trivial and type(rule.substep).__name__ == "AddRule":
            ctx["com_add"] = "éléments simples : $" + sp.latex(sp.simplify(g)) + "=" + sp.latex(new) + "$"
        elif not trivial:
            rows.append((wrap(_iq(new, var, A, B)), "on réécrit $" + sp.latex(sp.simplify(g)) + "=" + sp.latex(new) + "$"))
        return _applique(rule.substep, new, var, A, B, wrap, rows, ctx, prof + 1, fus)
    if nom == "PartsRule":
        u = sb(rule.u)
        dv = sb(rule.dv)
        du = sp.simplify(sp.diff(u, var))
        vv = _usuelle_q(dv, var)
        if vv is None:
            raise _Abandon()
        h = sp.simplify(du * vv)
        neg = False
        try:
            neg = bool(h.could_extract_minus_sign())
        except Exception:
            neg = False
        hd = sp.simplify(-h) if neg else h
        sgn = "+" if neg else "-"
        cyc = _detecte_cyclique(g, var, u, dv, vv, h)
        if cyc is not None:
            return _rediger_cyclique(g, var, A, B, wrap, rows, ctx, u, dv, vv, cyc)
        nu, nv = (("p", "q") if getattr(var, "name", "") in ("u", "v", "p", "q") else ("u", "v"))
        base = _crochet(sp.simplify(u * vv), A, B)
        rows.append((wrap(_joint(base, sgn, _iq(hd, var, A, B))),
                     "IPP : $" + nu + "=" + sp.latex(u) + "$, $" + nv + "'=" + sp.latex(dv) + "$ ; $" + nu + "'=" + sp.latex(du) + "$, $" + nv + "=" + _aff(vv) + "$"))
        from sympy.integrals.manualintegrate import integral_steps as _is
        sec = _avec_delai(lambda: _is(hd, var), 10)
        if sec is None:
            raise _Abandon()
        G2 = _applique(sec, hd, var, A, B, lambda L: wrap(_joint(base, sgn, L)), rows, ctx, prof + 1, True)
        G = sp.simplify(u * vv) + (G2 if neg else -G2)
        ap = wrap(_crochet(G, A, B))
        if not fus and (not rows or rows[-1][0] != ap):
            rows.append((ap, "on regroupe"))
        ctx["monde"] = (var, A, B, G)
        return G
    if nom == "URule":
        uf = sb(rule.u_func)
        if _affine(uf, var):
            if "PartsRule(" in repr(rule.substep):
                raise _Abandon()
            Gx = _usuelle_q(g, var)
            return terminal(Gx, "primitive usuelle")
        t = _nouveau_symbole(["t", "s", "w", "z"], list(g.free_symbols) + [var, ctx.get("X")])
        if not _croissant_sur(uf, var, A, B, ctx):
            raise _Abandon()
        try:
            clA, uA = _lim(uf, var, A, "+") if A in (sp.oo, -sp.oo) else ("fin", sp.simplify(uf.subs(var, A)))
            clB, uB = _lim(uf, var, B, "-") if B in (sp.oo, -sp.oo) else ("fin", sp.simplify(uf.subs(var, B)))
        except Exception:
            raise _Abandon()
        if clA not in ("fin", "inf") or clB not in ("fin", "inf"):
            raise _Abandon()
        sous = rule.substep
        sous_i = sous
        while type(sous_i).__name__ == "PiecewiseRule":
            br = [sub_ for sub_, cond_ in sous_i.subfunctions if (cond_ == True or sp.simplify(cond_) == True)]
            if not br:
                raise _Abandon()
            sous_i = br[0]
        gt = sp.simplify(sous_i.integrand.subs(rule.u_var, t))
        duf = sp.simplify(sp.diff(uf, var))
        rows.append((wrap(_iq(gt, t, uA, uB)),
                     "$t=" + sp.latex(uf) + "$, $dt=" + sp.latex(duf) + "\\,d" + sv + "$ ; $t$ est $C^1$ et strictement monotone"))
        # rule.substep est exprimée en rule.u_var
        sous2 = sous
        Gt = _applique(sous2, gt, t, uA, uB, wrap, rows, ctx, prof + 1)
        return Gt.subs(t, uf)
    if nom == "CyclicPartsRule":
        pr = rule.parts_rules
        if len(pr) != 2:
            raise _Abandon()
        u1, dv1 = sb(pr[0].u), sb(pr[0].dv)
        v1 = _usuelle_q(dv1, var)
        if v1 is None:
            raise _Abandon()
        cyc = _detecte_cyclique(g, var, u1, dv1, v1, sp.simplify(sp.diff(u1, var) * v1))
        if cyc is None:
            raise _Abandon()
        return _rediger_cyclique(g, var, A, B, wrap, rows, ctx, u1, dv1, v1, cyc)
    # règle terminale : primitive usuelle (liste blanche : fonctions élémentaires uniquement)
    if nom in _INTERDITS:
        raise _Abandon()
    if nom not in ("ConstantRule", "PowerRule", "NestedPowRule", "TrigRule", "SinRule", "CosRule", "SecTanRule", "CscCotRule", "Sec2Rule", "Csc2Rule",
                   "HyperbolicRule", "SinhRule", "CoshRule", "ExpRule", "ReciprocalRule", "ArcsinRule", "ArcsinhRule", "ReciprocalSqrtQuadraticRule",
                   "SqrtQuadraticDenomRule", "SqrtQuadraticRule", "CompleteSquareRule", "TrigSubstitutionRule", "ArctanRule", "DontKnowRule"):
        raise _Abandon()
    G = _usuelle_q(g, var)
    if G is not None and G.has(*_SPECF):
        raise _Abandon()
    return terminal(G, "primitive usuelle")


def _usuelle_q(g, var):
    G = _usuelle(g, var)
    return None if G is None else _logabs(G)


def _cond_lim_phrase(G, val):
    try:
        if len(G.atoms(sp.log)) >= 2 and not G.has(sp.atan):
            return "on regroupe les logarithmes : $\\ln a-\\ln b=\\ln\\frac{a}{b}$, puis continuité de $\\ln$"
        if G.has(sp.atan) or G.has(sp.acot):
            return "$\\arctan t\\to\\pm\\frac{\\pi}{2}$ à l'infini"
        if G.has(sp.exp) and G.has(sp.sin, sp.cos):
            return "fonction bornée fois une exponentielle de limite nulle"
        if G.has(sp.exp):
            return "croissances comparées"
        if G.has(sp.log):
            return "croissances comparées"
    except Exception:
        pass
    return "passage à la limite"


def _chaine_int(f, x, u, v, pb_u, pb_v, val, ordre=None):
    """Bloc aligné du calcul de ∫_u^v f dx ; None si le calcul ne se rédige pas proprement.
    pb_u / pb_v : la borne est un point à problème (intégrale impropre de ce côté). val : valeur déjà établie (peut valoir ±oo)."""
    try:
        if f.has(sp.Piecewise, sp.Abs):
            return _chaine_morceaux(f, x, u, v, pb_u, pb_v, val)
        from sympy.integrals.manualintegrate import integral_steps
        ctx = dict(X=None, cible=None, monde=None)
        rows = []
        X = None
        A, B = u, v
        wrap0 = lambda L: L
        debut = r"\int_{" + _borne_lat(u) + "}^{" + _borne_lat(v) + "}" + sp.latex(f) + r"\,d" + sp.latex(x)
        if pb_u and pb_v:
            return None
        if pb_u or pb_v:
            X = _nouveau_symbole(["X", "Y", "R", "Z"], list(f.free_symbols) + [x])
            ctx["X"] = X
            if pb_v:
                B, cible, sens = X, v, "-"
            else:
                A, cible, sens = X, u, "+"
            ctx["cible"] = cible
            lim_txt = r"\lim_{" + sp.latex(X) + r"\to " + _borne_lat(cible) + (r"^{-}" if sens == "-" and cible not in (sp.oo, -sp.oo) else (r"^{+}" if sens == "+" and cible not in (sp.oo, -sp.oo) else "")) + "}"
            wrap0 = lambda L: lim_txt + r"\left(" + L + r"\right)"
            rows.append((wrap0(_iq(f, x, A, B)), "définition de l'intégrale impropre"))
        regle = _avec_delai(lambda: integral_steps(f, x), 12)
        if regle is None:
            return None
        n0 = len(rows)
        G = _applique(regle, f, x, A, B, wrap0, rows, ctx)
        monde = ctx.get("monde")
        if monde is None:
            return None
        var_w, A_w, B_w, G_w = monde
        # cohérence : G est une primitive de f
        d = sp.simplify(sp.diff(G_w.subs(var_w, x) if var_w == x else G, x) - f)
        if d != 0:
            ok = False
            try:
                autres = {s_: sp.Rational(3, 2) for s_ in d.free_symbols if s_ != x}
                dd = d.subs(autres)
                ok = all(abs(complex(sp.N(dd.subs(x, t_)))) < 1e-9 for t_ in _echantillons_dans(u, v))
            except Exception:
                ok = False
            if not ok:
                return None
        # évaluation
        def ev(P, pt, sens):
            if pt in (sp.oo, -sp.oo):
                return None
            return sp.simplify(P.subs(var_w, pt))
        GB = G_w.subs(var_w, B_w)
        GA = G_w.subs(var_w, A_w)
        if X is None:
            def borne_val(P, pt, sens):
                try:
                    vv = sp.simplify(P.subs(var_w, pt))
                    if not vv.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                        return _par_ev(P.subs(var_w, pt)), vv
                except Exception:
                    pass
                cl_, lv_ = _lim(P, var_w, pt, sens)
                if cl_ != "fin":
                    raise _Abandon()
                return r"\lim_{" + sp.latex(var_w) + r"\to " + _borne_lat(pt) + ("^{+}" if sens == "+" else "^{-}") + r"}" + _par_ev(P), sp.simplify(lv_)
            if B_w in (sp.oo, -sp.oo) or A_w in (sp.oo, -sp.oo):
                return None
            tB, eB = borne_val(G_w, B_w, "-")
            tA, eA = borne_val(G_w, A_w, "+")
            diff_txt = _joint(tB, "-", tA)
            rows.append((diff_txt, "$f$ continue sur $[" + sp.latex(u) + "," + sp.latex(v) + "]$ : théorème fondamental" if "lim" not in diff_txt else "$f$ se prolonge par continuité aux bornes : théorème fondamental"))
            vf = sp.simplify(eB - eA)
            if val is not None and val not in (sp.oo, -sp.oo) and sp.simplify(vf - val) != 0:
                return None
            r2 = sp.latex(eB) if eA == 0 else _joint(_par_ev(eB), "-", _par_ev(eA))
            if r2 != diff_txt:
                rows.append((r2, ""))
            fin = val if (val is not None and val not in (sp.oo, -sp.oo)) else vf
            rf = sp.latex(fin)
            if rf != r2 and (("\\left(" in r2) or fin != (eB - eA)):
                rows.append((rf, ""))
        else:
            # impropre : la borne mobile est X
            GX = G_w.subs(var_w, X) if (A_w == X or B_w == X) else None
            if GX is None:
                # après changement de variable la borne mobile est une expression en X
                pass
            fixe = A_w if B_w.has(X) else B_w
            mob = B_w if B_w.has(X) else A_w
            if fixe in (sp.oo, -sp.oo):
                return None
            Gmob = G_w.subs(var_w, mob)
            Gfix = sp.simplify(G_w.subs(var_w, fixe))
            if B_w.has(X):
                interne = _joint(_par_ev(Gmob), "-", _par_ev(Gfix))
                interne_s = interne
            else:
                interne = _joint(_par_ev(Gfix), "-", _par_ev(Gmob))
            rows.append((wrap0(interne), "$f$ continue sur $[" + _borne_lat(A) + "," + _borne_lat(B) + "]$ : théorème fondamental" if (A not in (sp.oo, -sp.oo) and B not in (sp.oo, -sp.oo)) else "$f$ continue sur l'intervalle : théorème fondamental"))
            cl, lv = _lim((Gmob - Gfix) if B_w.has(X) else (Gfix - Gmob), X, ctx["cible"], "-" if pb_v else "+")
            if cl == "fin":
                lv = sp.simplify(lv)
                if val is not None and val not in (sp.oo, -sp.oo) and sp.simplify(lv - val) != 0:
                    return None
                rows.append((sp.latex(lv), _cond_lim_phrase(G_w, lv)))
            elif cl == "inf":
                if val is not None and lv != val:
                    return None
                rows.append((_lat(lv), "l'intégrale diverge"))
            else:
                return None
        return _bloc_chaine(debut, rows)
    except _Abandon:
        return None
    except Exception:
        return None


def _par_ev(e):
    s = sp.latex(e)
    if getattr(e, "is_Add", False):
        return r"\left(" + s + r"\right)"
    try:
        if e.could_extract_minus_sign():
            return r"\left(" + s + r"\right)"
    except Exception:
        pass
    return s


# -------------------------------------------------------------
# Fonctions spéciales exclues des primitives + intégrales de type Gauss (sans fonction d'erreur)
# -------------------------------------------------------------
_SPECF = (sp.erf, sp.erfc, sp.erfi, sp.Si, sp.Ci, sp.Shi, sp.Chi, sp.Ei, sp.li, sp.fresnels, sp.fresnelc, sp.expint,
          sp.lowergamma, sp.uppergamma, sp.polylog, sp.lerchphi, sp.hyper, sp.meijerg)

_GAUSS_Q = []


def _forme_gauss_q(g, v):
    """(C, P, a, b) si g = C * P(v) * exp(-a v^2 + b v) : a>0 et b numériques, P polynôme en v, C indépendant de v."""
    try:
        C, h = g.as_independent(v, as_Add=False)
        expo = sp.Integer(0)
        reste = sp.Integer(1)
        for fac in sp.Mul.make_args(h):
            if isinstance(fac, sp.exp):
                expo += fac.args[0]
            elif fac.is_Pow and fac.base == sp.E:
                expo += fac.exp
            else:
                reste *= fac
        if not expo.has(v):
            return None
        Q = sp.Poly(sp.expand(expo), v)
        if Q.degree() != 2:
            return None
        c2, c1, c0 = Q.all_coeffs()
        if not (c2.is_number and c1.is_number and c2 < 0):
            return None
        P = sp.Poly(sp.expand(reste), v)
        return (sp.simplify(C * sp.exp(c0)), P.as_expr(), -c2, c1)
    except Exception:
        return None


def _gauss_valeur_q(g, v, lo, hi):
    fg = _forme_gauss_q(g, v)
    if fg is None:
        return None
    C, P, a, b = fg
    try:
        if b == 0 and all(j % 2 == 1 for (j,), _c in sp.Poly(P, v).terms()):
            return None
        if lo == -sp.oo and hi == sp.oo:
            c = sp.simplify(b / (2 * a))
            w = sp.Dummy("w")
            Pw = sp.Poly(sp.expand(P.subs(v, w + c)), w)
            tot = sp.Integer(0)
            for (j,), coef in Pw.terms():
                if j % 2 == 0:
                    tot += coef * sp.gamma(sp.Rational(j + 1, 2)) / a ** sp.Rational(j + 1, 2)
            return sp.simplify(C * sp.exp(b ** 2 / (4 * a)) * tot)
        if lo == 0 and hi == sp.oo and b == 0:
            Pv = sp.Poly(sp.expand(P), v)
            tot = sp.Integer(0)
            for (j,), coef in Pv.terms():
                tot += coef * sp.gamma(sp.Rational(j + 1, 2)) / (2 * a ** sp.Rational(j + 1, 2))
            return sp.simplify(C * tot)
    except Exception:
        return None
    return None


def _bloc_lemme_gauss():
    """Bloc aligné du lemme de Gauss (affiché une seule fois par étude)."""
    lignes = [
        (r"\left(\int_{\mathbb{R}}e^{-s^{2}}ds\right)\left(\int_{\mathbb{R}}e^{-t^{2}}dt\right)", ""),
        (r"\iint_{\mathbb{R}^{2}}e^{-(s^{2}+t^{2})}\,ds\,dt", "Tonelli : intégrande positive"),
        (r"\int_{0}^{2\pi}\!\int_{0}^{+\infty}r\,e^{-r^{2}}\,dr\,d\theta", "coordonnées polaires, jacobien $r$"),
        (r"2\pi\left[-\tfrac{1}{2}e^{-r^{2}}\right]_{0}^{+\infty}", "Tonelli, puis primitive de $re^{-r^{2}}$"),
        (r"\pi", ""),
    ]
    b = _bloc_chaine("G^{2}", lignes)
    return b


def _chaine_gauss(g, v, lo, hi, val):
    """Blocs LaTeX (liste) du calcul d'une intégrale de type Gauss sur R ou [0,+oo[ ; None si non applicable."""
    try:
        fg = _forme_gauss_q(g, v)
        if fg is None:
            return None
        vx = _gauss_valeur_q(g, v, lo, hi)
        if vx is None:
            return None
        if val is not None and sp.simplify(vx - val) != 0:
            return None
        C, P, a, b = fg
        sv = sp.latex(v)
        debut = r"\int_{" + _borne_lat(lo) + "}^{" + _borne_lat(hi) + "}" + sp.latex(g) + r"\,d" + sv
        blocs = []
        if "lemme" not in _GAUSS_Q:
            _GAUSS_Q.append("lemme")
            blocs.append(_T("Lemme de Gauss : on note $G=\\int_{\\mathbb{R}}e^{-t^{2}}dt>0$."))
            blocs.append(_bloc_lemme_gauss())
            blocs.append(_T("Donc $G^{2}=\\pi$ et, comme $G>0$, $G=\\sqrt{\\pi}$. Pour $a>0$ le changement $t=\\sqrt{a}\\,w$ donne $\\int_{\\mathbb{R}}e^{-aw^{2}}dw=\\sqrt{\\pi/a}$."))
        rows = []
        sa = sp.latex(a)
        if lo == -sp.oo:
            c = sp.simplify(b / (2 * a))
            d = sp.simplify(b ** 2 / (4 * a))
            w = sp.Symbol("w", real=True)
            if b != 0:
                Pw = sp.expand(P.subs(v, w + c))
                pref = sp.simplify(C * sp.exp(d))
                rows.append((_mulc(pref, r"\int_{\mathbb{R}}\left(" + sp.latex(Pw) + r"\right)e^{-" + ("" if a == 1 else sa) + r"w^{2}}\,dw") if sp.Poly(Pw, w).degree() > 0 else _mulc(pref, r"\int_{\mathbb{R}}e^{-" + ("" if a == 1 else sa) + r"w^{2}}\,dw"),
                             "$-" + ("" if a == 1 else sa) + sv + "^{2}+" + ("" if b == 1 else sp.latex(b)) + sv + "=-" + ("" if a == 1 else sa) + "(" + sv + "-" + sp.latex(c) + ")^{2}+" + sp.latex(d) + "$ et $w=" + sv + "-" + sp.latex(c) + "$ (translation, mesure de Lebesgue invariante)"))
            else:
                Pw = sp.expand(P)
                w = v
                pref = sp.simplify(C)
            Pp = sp.Poly(Pw, w)
            pairs = [(j, cf) for (j,), cf in Pp.terms() if j % 2 == 0]
            impairs = [j for (j,), cf in Pp.terms() if j % 2 == 1]
            if impairs:
                Pe = sum(cf * w ** j for j, cf in pairs)
                rows.append((_mulc(pref, r"\int_{\mathbb{R}}\left(" + sp.latex(Pe) + r"\right)e^{-" + ("" if a == 1 else sa) + sp.latex(w) + r"^{2}}\,d" + sp.latex(w)) if Pe != 0 else "0",
                             "les termes de degré impair ont une intégrande impaire et intégrable : intégrale nulle"))
            if pairs and any(j > 0 for j, _c in pairs):
                termes = []
                for j, cf in pairs:
                    m = sp.factorial2(2 * (j // 2) - 1) if j > 0 else 1
                    mom = (r"\frac{%s}{%s}" % (sp.latex(m), sp.latex((2 * a) ** (j // 2)))) if (m != 1 or (2 * a) ** (j // 2) != 1) else ""
                    rac = r"\sqrt{\pi}" if a == 1 else r"\sqrt{\frac{\pi}{%s}}" % sa
                    termes.append(_mulc(cf, mom + rac) if cf != 1 else mom + rac)
                expr = termes[0]
                for t_ in termes[1:]:
                    expr = _joint(expr, "+", t_)
                rows.append((_mulc(pref, r"\left(" + expr + r"\right)") if len(termes) > 1 else _mulc(pref, expr),
                             "$\\int_{\\mathbb{R}}w^{2j}e^{-aw^{2}}dw=\\frac{(2j-1)!!}{(2a)^{j}}\\sqrt{\\pi/a}$ (IPP répétée)"))
            elif pairs:
                cf = pairs[0][1]
                rac = r"\sqrt{\pi}" if a == 1 else r"\sqrt{\frac{\pi}{%s}}" % sa
                rows.append((_mulc(pref * cf, rac), "$\\int_{\\mathbb{R}}e^{-aw^{2}}dw=\\sqrt{\\pi/a}$"))
            rows.append((sp.latex(vx), ""))
        else:
            Pp = sp.Poly(sp.expand(P), v)
            pairs = [(j, cf) for (j,), cf in Pp.terms()]
            termes = []
            for j, cf in pairs:
                if j % 2 == 0:
                    m = sp.factorial2(j - 1) if j > 0 else 1
                    r_ = sp.Rational(1, 2) * m / (2 * a) ** (j // 2)
                    rac = r"\sqrt{\pi}" if a == 1 else r"\sqrt{\frac{\pi}{%s}}" % sa
                    termes.append(_mulc(cf * r_, rac))
                else:
                    i_ = (j - 1) // 2
                    termes.append(_mulc(cf, r"\frac{%s}{%s}" % (sp.latex(sp.factorial(i_)), sp.latex(2 * a ** (i_ + 1)))))
            expr = termes[0]
            for t_ in termes[1:]:
                expr = _joint(expr, "+", t_)
            rows.append((_mulc(sp.simplify(C), r"\left(" + expr + r"\right)") if len(termes) > 1 else _mulc(sp.simplify(C), expr),
                         "parité : $\\int_{0}^{+\\infty}w^{2j}e^{-aw^{2}}=\\frac12\\int_{\\mathbb{R}}$ ; $\\int_{0}^{+\\infty}w^{2i+1}e^{-aw^{2}}=\\frac{i!}{2a^{i+1}}$ ($u=w^{2}$)"))
            rows.append((sp.latex(vx), ""))
        # première ligne : égalité avec elle-même omise ; le bloc commence par l'intégrale
        if len(rows) >= 2 and rows[-1][0] == rows[-2][0]:
            rows.pop()
        blocs.append(_bloc_chaine(debut, rows))
        return blocs
    except Exception:
        return None


def _chaine_morceaux(f, x, u, v, pb_u, pb_v, val):
    """Intégrale d'une fonction définie par morceaux / avec valeur absolue (bornes finies, propre) : découpage puis primitives usuelles."""
    try:
        if pb_u or pb_v or u in (sp.oo, -sp.oo) or v in (sp.oo, -sp.oo):
            return None
        inter = set(q for q in _points_raccord(f, x, u, v) if _num(u) < _num(q) < _num(v))
        pts = [u] + sorted(inter, key=lambda q: _num(q)) + [v]
        if len(pts) < 3:
            return None
        morceaux = []
        for a, b in zip(pts[:-1], pts[1:]):
            e = _expr_sur(f, x, a, b)
            G = _usuelle_q(e, x)
            if G is None or G.has(*_SPECF):
                return None
            morceaux.append((a, b, e, G))
        debut = r"\int_{" + _borne_lat(u) + "}^{" + _borne_lat(v) + "}" + sp.latex(f) + r"\,d" + sp.latex(x)
        def somme(lst):
            base = lst[0]
            for m_ in lst[1:]:
                base = _joint(base, "+", m_)
            return base
        desc = " ; ".join("$f(" + sp.latex(x) + ")=" + sp.latex(e) + "$ sur $[" + _borne_lat(a) + "," + _borne_lat(b) + "]$" for a, b, e, _G in morceaux)
        rows = [(somme([_iq(e, x, a, b) for a, b, e, _G in morceaux]), "relation de Chasles ; " + desc)]
        rows.append((somme([_crochet(G, a, b) for a, b, e, G in morceaux]), "primitives usuelles"))
        vals = [sp.simplify(G.subs(x, b) - G.subs(x, a)) for a, b, e, G in morceaux]
        rows.append((somme([sp.latex(vv) if True else "" for vv in vals]) if all(vv.is_number for vv in vals) else somme([sp.latex(vv) for vv in vals]), "théorème fondamental sur chaque segment"))
        tot = sp.simplify(sum(vals, sp.Integer(0)))
        if val is not None and val not in (sp.oo, -sp.oo) and sp.simplify(tot - val) != 0:
            return None
        rows.append((sp.latex(tot), ""))
        return _bloc_chaine(debut, rows)
    except Exception:
        return None




# -------------------------------------------------------------
# Géométrie d'une demi-intégrale (entre un point régulier c et un bord à problème p)
# -------------------------------------------------------------
def _nom_borne(p):
    return r"+\infty" if p == sp.oo else (r"-\infty" if p == -sp.oo else sp.latex(p))


def _p_plus_grand(p, c):
    if p == sp.oo:
        return True
    if p == -sp.oo:
        return False
    return _num(p) > _num(c)


def _ordre(c, p):
    """(lo, hi, lo_ouvert, hi_ouvert) de l'intervalle d'extrémités c (fermée) et p (ouverte)."""
    if _p_plus_grand(p, c):
        return c, p, False, True
    return p, c, True, False


def _int_lat(c, p, integrande="f(x)"):
    lo, hi, _a, _b = _ordre(c, p)
    return r"\int_{" + _borne_lat(lo) + r"}^{" + _borne_lat(hi) + r"}" + integrande + r"\,dx"


def _interv_demi(c, p):
    lo, hi, ol, oh = _ordre(c, p)
    return _iv(lo, hi, ol, oh)


def _pt_txt(c, p):
    """Voisinage de p, vu depuis c : +\\infty, -\\infty, 0^{+}, 1^{-}."""
    if p in (sp.oo, -sp.oo):
        return _nom_borne(p)
    return sp.latex(p) + (r"^{+}" if not _p_plus_grand(p, c) else r"^{-}")


def _sens(c, p):
    return "-" if _p_plus_grand(p, c) else "+"


def _facteur_oscillant(f, x):
    """f = h(x) * (produit de sin/cos) : renvoie (h, osc, k) ; k = pulsation si un seul facteur, None sinon."""
    try:
        facteurs = sp.Mul.make_args(sp.factor(f))
        osc = [t for t in facteurs if t.func in (sp.sin, sp.cos) and t.has(x)]
        if not osc:
            return None
        o = sp.Mul(*osc)
        h = sp.simplify(f / o)
        if h.has(sp.sin) or h.has(sp.cos):
            return None
        if len(osc) == 1:
            k = sp.diff(osc[0].args[0], x)
            if k.has(x) or k == 0:
                return None
            return h, osc[0], k
        return h, o, None
    except Exception:
        return None


# -------------------------------------------------------------
# Rédaction des critères de comparaison
# -------------------------------------------------------------
def _dist(x, p, c):
    """Distance signée à p, du côté d'où l'on approche p (LaTeX, avec parenthèses si nécessaire)."""
    d = sp.latex(sp.simplify(p - x)) if _p_plus_grand(p, c) else sp.latex(sp.simplify(x - p))
    return d


def _unilat(l, x, p, c):
    """Remplace |x-p| par x-p ou p-x (selon le côté d'où l'on approche p) et précise le côté de la limite."""
    if p in (sp.oo, -sp.oo):
        return l
    d_abs = sp.latex(sp.Abs(sp.simplify(x - p)))
    d = _dist(x, p, c)
    d_par = d if d == "x" else "(" + d + ")"
    l = l.replace("{" + d_abs + "}", "{" + d + "}").replace(d_abs, d_par)
    cote = r"^{-}" if _p_plus_grand(p, c) else r"^{+}"
    return l.replace(r"x\to " + sp.latex(p) + "}", r"x\to " + sp.latex(p) + cote + "}")


def _equiv_abs(eq, x, p, c, nom="f"):
    """Équivalent de |nom(x)| en p (LaTeX)."""
    l1, _ = _texte_equivalent(eq, x, p, 1)
    return _unilat(l1, x, p, c).replace("f(x)", r"|" + nom + r"(x)|", 1)


def _ref(eq, x, p, c):
    """Fonction de référence (membre de droite de l'équivalent)."""
    l1, _ = _texte_equivalent(eq, x, p, 1)
    return _unilat(l1, x, p, c).split(r"{\sim}\ ", 1)[1]


_ROT = {}
_REFS = set()


def _V(cle, *options):
    """Variante de rédaction : tourne parmi les options pour éviter les formules identiques."""
    i = _ROT.get(cle, 0)
    _ROT[cle] = i + 1
    return options[i % len(options)]


def _premiere(cle):
    """Vrai la première fois seulement (pour n'énoncer un critère de référence qu'une fois)."""
    if cle in _REFS:
        return False
    _REFS.add(cle)
    return True


def _note_log(gam, alpha):
    if gam == 0 or alpha == 1 or not _premiere("notelog"):
        return ""
    return " Le facteur logarithmique ne joue aucun rôle ici, car $\\alpha\\ne1$ : il est négligeable devant toute puissance."


def _justif_equiv(eq, x, p, c, nom="|f|"):
    """Justification brève d'un équivalent : renvoie (texte LaTeX, verdict). La conclusion porte sur l'intégrale de `nom`."""
    v = _verdict_equivalent(eq, p)
    e, gam = eq["e"], eq["gamma"]
    inf = p in (sp.oo, -sp.oo)
    I = _int_lat(c, p, "|" + nom.strip("|") + "(x)|")
    cv = _V("cv", "$|" + nom.strip("|") + "|$ étant positive et équivalente à cette référence, $" + I + "$ converge." if _premiere("th_eq_c") else "donc $" + I + "$ converge.", "d'où la convergence de $" + I + "$.", "et $" + I + "$ converge par équivalence.", "ce qui assure la convergence de $" + I + "$.")
    dv = _V("dv", "$|" + nom.strip("|") + "|$ étant positive et équivalente à cette référence, $" + I + "$ diverge." if _premiere("th_eq_d") else "donc $" + I + "$ diverge.", "d'où la divergence de $" + I + "$.", "et $" + I + "$ diverge par équivalence.", "ce qui entraîne la divergence de $" + I + "$.")
    if inf and gam == 0 and sp.simplify(-e) <= 0:
        C = sp.latex(eq["C"])
        if sp.simplify(-e) == 0:
            return _T("Autrement dit $|f(x)|\\to " + C + ">0$ : dès que $x$ est assez grand, $|f|\\ge\\frac{" + C + "}{2}$ sur un intervalle de longueur infinie, et une constante non nulle n'y est pas intégrable ; donc $" + I + "$ diverge."), False
        return _T("Autrement dit $|f(x)|\\to+\\infty$ : on a $|f|\\ge1$ sur un intervalle de longueur infinie, ce qui est incompatible avec l'intégrabilité ; donc $" + I + "$ diverge."), False
    if inf:
        alpha = sp.simplify(-e)
        var = "x" if p == sp.oo else "|x|"
        borne = r"^{+\infty}" if p == sp.oo else r"_{-\infty}"
        if gam != 0 and alpha == 1:
            beta = -gam
            if _premiere("bertrand_inf"):
                return _T("Or $\\int" + borne + r"\frac{dx}{" + var + r"(\ln " + var + r")^{\beta}}$ converge si et seulement si $\beta>1$ ; ici $\beta=" + sp.latex(beta) + "$" + (", " if v else ", ") + (cv if v else dv)), v
            return _T(_V("be", "Avec $\\beta=" + sp.latex(beta) + "$ " + ("(>1)" if v else "(\\le1)") + ", ", "Le même critère de Bertrand donne ici $\\beta=" + sp.latex(beta) + "$, ") + (cv if v else dv)), v
        if _premiere("riemann_inf"):
            return _T("Or $\\int" + borne + r"\frac{dx}{" + var + r"^{\alpha}}$ converge si et seulement si $\alpha>1$ ; ici $\alpha=" + sp.latex(alpha) + "$, " + (cv if v else dv) + _note_log(gam, alpha)), v
        ineq = ">1" if v else "\\le1"
        if gam == 0 and _V("form_i", 0, 1) == 1:
            Cc = sp.latex(eq["C"])
            if v:
                return _T("Comme $" + var + "^{" + sp.latex(alpha) + "}|f(x)|\\to " + Cc + "$, on a $|f(x)|\\le\\frac{" + sp.latex(sp.simplify(eq["C"] + 1)) + "}{" + var + "^{" + sp.latex(alpha) + "}}$ pour $|x|$ grand ; avec $\\alpha=" + sp.latex(alpha) + ">1$, $" + I + "$ converge par comparaison."), v
            return _T("Comme $" + var + "^{" + sp.latex(alpha) + "}|f(x)|\\to " + Cc + ">0$, on a $|f(x)|\\ge\\frac{" + sp.latex(sp.simplify(eq["C"] / 2)) + "}{" + var + "^{" + sp.latex(alpha) + "}}$ pour $|x|$ grand ; avec $\\alpha=" + sp.latex(alpha) + "\\le1$, $" + I + "$ diverge par comparaison."), v
        return _T(_V("ri", "L'exposant est $\\alpha=" + sp.latex(alpha) + ineq + "$ : ", "Ici $\\alpha=" + sp.latex(alpha) + ineq + "$, ", "Puisque $\\alpha=" + sp.latex(alpha) + ineq + "$, ", "Avec $\\alpha=" + sp.latex(alpha) + ineq + "$, ")
                  + (cv if v else dv) + _note_log(gam, alpha)), v
    d = _dist(x, p, c)
    if d != "x":
        d = "(" + d + ")"
    if gam != 0 and e == 1:
        beta = -gam
        if _premiere("bertrand_fin"):
            return _T("Or $\\int\\frac{dx}{" + d + r"\,|\ln " + d + r"|^{\beta}}$ converge au voisinage de $" + sp.latex(p) + "$ si et seulement si $\\beta>1$ ; ici $\\beta=" + sp.latex(beta) + "$, " + (cv if v else dv)), v
        return _T(_V("bf", "Avec $\\beta=" + sp.latex(beta) + ("$ (>1), " if v else "$ (\\le1), "), "Le critère de Bertrand donne de même $\\beta=" + sp.latex(beta) + "$ : ") + (cv if v else dv)), v
    if _premiere("riemann_fin"):
        return _T("Or $\\int\\frac{dx}{" + d + r"^{\alpha}}$ converge au voisinage de $" + sp.latex(p) + "$ si et seulement si $\\alpha<1$ ; ici $\\alpha=" + sp.latex(e) + "$, " + (cv if v else dv) + _note_log(gam, e)), v
    ineq = "<1" if v else "\\ge1"
    if gam == 0 and _V("form_f", 0, 1) == 1:
        Cc = sp.latex(eq["C"])
        if v:
            return _T("Comme $" + d + "^{" + sp.latex(e) + "}|f(x)|\\to " + Cc + "$, on a $|f(x)|\\le\\frac{" + sp.latex(sp.simplify(eq["C"] + 1)) + "}{" + d + "^{" + sp.latex(e) + "}}$ près de $" + sp.latex(p) + "$ ; avec $\\alpha=" + sp.latex(e) + "<1$, $" + I + "$ converge par comparaison."), v
        return _T("Comme $" + d + "^{" + sp.latex(e) + "}|f(x)|\\to " + Cc + ">0$, on a $|f(x)|\\ge\\frac{" + sp.latex(sp.simplify(eq["C"] / 2)) + "}{" + d + "^{" + sp.latex(e) + "}}$ près de $" + sp.latex(p) + "$ ; avec $\\alpha=" + sp.latex(e) + "\\ge1$, $" + I + "$ diverge par comparaison."), v
    return _T(_V("rf", "L'exposant est $\\alpha=" + sp.latex(e) + ineq + "$ : ", "Ici $\\alpha=" + sp.latex(e) + ineq + "$, ", "Puisque $\\alpha=" + sp.latex(e) + ineq + "$, ", "Avec $\\alpha=" + sp.latex(e) + ineq + "$, ")
              + (cv if v else dv) + _note_log(gam, e)), v


def _lead(pt, c, p):
    return _V("lead", "Près de $" + pt + "$, ", "Au voisinage de $" + pt + "$, ", "Quand $x\\to " + pt + "$, ", "À l'approche de $" + pt + "$, ")


def _texte_dominee(genre, nom, x, p, I, croiss, debut=True):
    """Domination par une référence intégrable ; la conclusion porte sur l'intégrale I."""
    if genre == "inf":
        lim = "$x^{2}|" + nom + "(x)|\\to0$ quand $x\\to" + _nom_borne(p) + "$"
        bnd = "$|" + nom + "(x)|\\le\\frac{1}{x^{2}}$ pour $|x|$ assez grand"
        ref = "$" + _nom_borne(p) + "$"
    else:
        d = sp.latex(sp.Abs(sp.simplify(x - p)))
        lim = "$" + d + r"^{1/2}|" + nom + "(x)|\\to0$ quand $x\\to" + sp.latex(p) + "$"
        bnd = "$|" + nom + "(x)|\\le\\frac{1}{" + d + r"^{1/2}}$ près de $" + sp.latex(p) + "$"
        ref = "$" + sp.latex(p) + "$"
    if _premiere("dom_" + genre):
        pre = ("par croissances comparées, " if croiss else "on a ")
        txt = pre + lim + ", donc " + bnd + " ; cette référence étant intégrable en " + ref + ", $" + I + "$ converge par comparaison."
    else:
        txt = _V("domc", "même argument de comparaison : " + lim + ", d'où " + bnd + ", et $" + I + "$ converge.", "la domination se reproduit : " + lim + " et " + bnd + ", donc $" + I + "$ converge.")
    if debut:
        txt = txt[0].upper() + txt[1:]
    return _T(txt)


# -------------------------------------------------------------
# Intégrabilité de f entre c et le bord à problème p
# -------------------------------------------------------------
def _abs_integrable(f, x, fm, c, p):
    """Étude de ∫|f| entre c et p. Renvoie (verdict True/False/None, lignes LaTeX)."""
    lignes = []
    pt = _pt_txt(c, p)
    I = _int_lat(c, p, "|f(x)|")
    croiss = bool(f.has(sp.exp) or f.has(sp.log))
    sg, tem = _signe_voisinage(fm, c, p)
    if sg in (1, -1):
        eq = _equivalent(sg * f, x, c, p)
        if eq is not None:
            lignes.append(_T(_lead(pt, c, p)) + " " + _equiv_abs(eq, x, p, c))
            texte, v = _justif_equiv(eq, x, p, c)
            lignes.append(texte)
            return v, lignes
        okd, genre = _domination(sg * f, x, c, p)
        if okd:
            lignes.append(_T(_lead(pt, c, p)) + " " + _texte_dominee(genre, "f", x, p, I, croiss, False))
            return True, lignes
    elif sg == 0:
        fo = _facteur_oscillant(f, x)
        if fo is not None and p in (sp.oo, -sp.oo):
            h, o, k = fo
            bs = chr(92)
            if not h.has(x):
                kk0 = abs(k) if k is not None else 1
                Ch = sp.latex(sp.Abs(h))
                lignes.append(_T(_lead(pt, c, p).rstrip() + " $f(x)=" + (sp.latex(h) + "\\," if h != 1 else "") + sp.latex(o) + "$ oscille entre signes opposés sans s'amortir. Sur chaque intervalle de longueur $" + sp.latex(sp.pi / kk0) + "$, $\\int|" + sp.latex(o)
                                 + "|=" + sp.latex(2 / kk0) + "$, donc chaque période ajoute à $\\int|f|$ la quantité fixe $" + sp.latex(sp.simplify(2 * sp.Abs(h) / kk0)) + "$ ; ces périodes sont en nombre infini, d'où $" + I + "=+\\infty$."))
                return False, lignes
            borne_txt = (r"|" + sp.latex(o) + r"|\le 1") if k is not None else r"\text{ chacun des facteurs trigonométriques étant majoré par } 1"
            lignes.append(_T(_lead(pt, c, p).rstrip() + " $f$ change de signe indéfiniment ; on écrit $f(x)=h(x)\\cdot " + sp.latex(o) + "$ avec $h(x)=" + sp.latex(h) + "$."))
            sh, _t = _signe_voisinage(_fnum(h, x), c, p)
            if sh in (1, -1):
                eq = _equivalent(sh * h, x, c, p)
                if eq is None:
                    okd, genre = _domination(sh * h, x, c, p)
                    if okd:
                        lignes.append(_texte_dominee(genre, "h", x, p, _int_lat(c, p, "|h(x)|"), croiss, True))
                        lignes.append((_T("Comme ") + " " + borne_txt + " " + _T(", on a $|f|\\le|h|$ : $" + I + "$ converge.")) if k is not None else (borne_txt.replace("\\text{ ", "\\text{").replace("} 1", " par 1}") + " " + _T("; ainsi $|f|\\le|h|$ et $" + I + "$ converge.")))
                        return True, lignes
                else:
                    lignes.append(_equiv_abs(eq, x, p, c, "h"))
                    texte, cv = _justif_equiv(eq, x, p, c, "|h|")
                    lignes.append(texte)
                    if cv:
                        lignes.append((_T("Comme ") + " " + borne_txt + " " + _T(", on a $|f|\\le|h|$ : par comparaison, $" + I + "$ converge.")) if k is not None else (_T("Chacun des facteurs trigonométriques étant majoré par $1$, $|f|\\le|h|$ : par comparaison, $" + I + "$ converge.")))
                        return True, lignes
                    if k is None:
                        pass
                    else:
                        u_ = sp.latex(o.args[0])
                        fonc = "sin" if o.func == sp.sin else "cos"
                        kk = abs(k)
                        alpha_h = -eq["e"]
                        if alpha_h < 0 or (alpha_h == 0 and eq["gamma"] == 0):
                            m_ = sp.latex(sp.simplify(eq["C"] / 2))
                            lignes.append(_T("Sur chaque intervalle de longueur $\\frac{\\pi}{" + sp.latex(kk) + "}$, $\\int|" + sp.latex(o) + "|=\\frac{2}{" + sp.latex(kk) + "}$ et $|h|\\ge " + m_ + ">0$ pour $|x|$ grand : chaque période ajoute à $\\int|f|$ au moins $\\frac{2\\cdot" + m_ + "}{" + sp.latex(kk) + "}$, quantité fixe. Ainsi $" + I + "=+\\infty$."))
                            return False, lignes
                        dev = r"\frac{1-\cos(2(" + u_ + r"))}{2}" if fonc == "sin" else r"\frac{1+\cos(2(" + u_ + r"))}{2}"
                        lignes.append(_T("Par ailleurs $|" + sp.latex(o) + "|\\ge " + bs + fonc + "^{2}(" + u_ + ")=" + dev + "$, d'où $|f|\\ge\\frac{|h|}{2}" + ("-" if fonc == "sin" else "+") + "\\frac{|h|\\cos(2(" + u_ + "))}{2}$."))
                        try:
                            hp = sp.simplify(sp.diff(h, x))
                            ipp = _T("Une intégration par parties ($h'(x)=" + sp.latex(hp) + "$, $h\\to0$ monotone, donc $\\int|h'|=|h(c)|$) montre que $\\int h\\cos(2(" + u_ + "))$ converge.")
                        except Exception:
                            ipp = _T("Une intégration par parties montre que $\\int h\\cos(2(" + u_ + "))$ converge.")
                        lignes.append(ipp)
                        lignes.append(_T("Comme $\\int\\frac{|h|}{2}$ diverge, $" + I + "=+\\infty$. Cela ne contredit pas la convergence de $\\int f$ : le critère d'Abel (avec $h$ décroissante de limite $0$ et une primitive de $" + sp.latex(o) + "$ bornée) l'assure sans valeur absolue."))
                        return False, lignes
        else:
            lignes.append(_T(_lead(pt, c, p).rstrip() + " $f$ change de signe indéfiniment."))
    # --- repli numérique
    mesures = []
    cf = float(sp.N(c))
    if p in (sp.oo, -sp.oo):
        sgn = 1 if p == sp.oo else -1
        bornes = [sgn * 10.0 ** k for k in (2, 3, 4)]
    else:
        pf = float(sp.N(p))
        bornes = [pf + (cf - pf) * 10.0 ** (-k) for k in (1, 2, 3)]
    for B in bornes:
        try:
            mp.mp.dps = 15
            v = _avec_delai(lambda: float(mp.quad(lambda t: abs(fm(t)), mp.linspace(mp.mpf(cf), mp.mpf(B), 41))), 20)
        except Exception:
            v = None
        mesures.append((B, v))
    if all(v is not None for _, v in mesures):
        vs = [v for _, v in mesures]
        incr = [vs[1] - vs[0], vs[2] - vs[1]]
        lignes.append(_T("Aucun équivalent simple de $|f|$ n'apparaît : on estime $\\int_{" + sp.latex(c) + "}^{X}|f|$ pour $X=" + ",\\ ".join(sp.latex(sp.Float(B, 5)) for B, _ in mesures)
                         + "$, soit $" + ",\\ ".join(sp.latex(sp.Float(v, 6)) for v in vs) + "$."))
        if incr[0] > 1e-9 and incr[1] < 0.35 * incr[0]:
            lignes.append(_T("Les accroissements $" + sp.latex(sp.Float(incr[0], 4)) + "\\to " + sp.latex(sp.Float(incr[1], 4)) + "$ s'écrasent : $\\int|f|$ semble converger (indication numérique, non démontrée)."))
            return True, lignes
        if incr[0] > 1e-9:
            lignes.append(_T("Les accroissements $" + sp.latex(sp.Float(incr[0], 4)) + "\\to " + sp.latex(sp.Float(incr[1], 4)) + "$ ne s'écrasent pas : $\\int|f|$ semble diverger (indication numérique, non démontrée)."))
            return False, lignes
    lignes.append(_T("Cette étude ne permet pas de conclure sur $\\int|f|$."))
    return None, lignes


# -------------------------------------------------------------
# Valeur de F en un point / une borne
# -------------------------------------------------------------
def _valeur_F(F, x, pt, sens):
    """(classe, valeur) de F en pt : substitution directe si possible, sinon limite."""
    if pt in (sp.oo, -sp.oo):
        return _lim(F, x, pt, None)
    try:
        v = sp.simplify(F.subs(x, pt))
        if v.is_finite is True and not v.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
            return "fin", v
    except Exception:
        pass
    return _lim(F, x, pt, sens)


def _pourquoi(F, x, pt, cl, v):
    """Brève justification d'une limite de F (ou chaîne vide)."""
    try:
        inf = pt in (sp.oo, -sp.oo)
        if cl == "fin" and inf and F.has(sp.atan) and _premiere("p_atan"):
            return " (car $\\arctan t\\to\\pm\\frac{\\pi}{2}$ quand $t\\to\\pm\\infty$)"
        if cl == "fin" and inf and F.has(sp.erf) and _premiere("p_erf"):
            return " (car $\\operatorname{erf}(t)\\to\\pm1$ quand $t\\to\\pm\\infty$)"
        if cl == "fin" and inf and F.has(sp.exp) and v == 0 and _premiere("p_exp"):
            return " (par croissances comparées)"
        if cl == "fin" and (not inf) and pt == 0 and F.has(sp.log) and _premiere("p_log"):
            return " (par croissances comparées : $t^{a}\\ln t\\to0$ quand $t\\to0^{+}$ pour $a>0$)"
    except Exception:
        pass
    return ""


def _fait_F(cl, v, pt, sens, probleme, F=None, x=None):
    """Phrase élémentaire sur F en pt (texte + LaTeX)."""
    if not probleme and cl == "fin":
        return "$F(" + sp.latex(pt) + ")=" + sp.latex(v) + "$"
    quand = "quand $" + _lim_txt(pt, sens) + "$"
    if cl == "fin":
        return "$F(x)\\to " + sp.latex(v) + "$ " + quand + (_pourquoi(F, x, pt, cl, v) if F is not None else "")
    if cl == "inf":
        return "$F(x)\\to " + _lat(v) + "$ " + quand
    if cl == "osc":
        return "$F$ n'a pas de limite " + quand + " (elle oscille)"
    return "la limite de $F$ " + quand + " n'a pas pu être déterminée"


# -------------------------------------------------------------
# Calcul de l'intégrale sur un morceau ]lo, hi[
# -------------------------------------------------------------
def _expr_sur(f, x, u, v):
    """f restreinte à ]u, v[ (branches de Piecewise choisies au milieu)."""
    mid = (u + v) / 2
    def _br(pw):
        for e_, c_ in pw.args:
            try:
                if bool(c_.subs(x, mid)):
                    return e_
            except Exception:
                pass
        return pw
    g = f.replace(lambda e: isinstance(e, sp.Piecewise), _br)

    def _ab(e):
        try:
            return e.args[0] if e.args[0].subs(x, mid) > 0 else -e.args[0]
        except Exception:
            return e
    return g.replace(lambda e: isinstance(e, sp.Abs), _ab)


def _points_raccord(f, x, lo, hi):
    pts = set()
    try:
        for pw in f.atoms(sp.Piecewise):
            for e_, ens in pw.as_expr_set_pairs():
                for bb in (getattr(ens, "start", None), getattr(ens, "end", None)):
                    if bb is not None and bb.is_finite and _num(bb) is not None and _num(lo) < _num(bb) < _num(hi):
                        pts.add(bb)
        for d_ in f.atoms(sp.Abs):
            sol = _avec_delai(lambda: sp.solveset(d_.args[0], x, domain=sp.Reals), 8)
            if isinstance(sol, sp.FiniteSet):
                for bb in sol.args:
                    if _num(bb) is not None and _num(lo) < _num(bb) < _num(hi):
                        pts.add(bb)
    except Exception:
        pass
    return pts


def _connue(f, x, lo, hi):
    """Phrase de reconnaissance d'une intégrale classique, ou None."""
    try:
        a_ = sp.Wild("a", exclude=[x])
        m = f.match(sp.exp(-a_ * x ** 2))
        if m and m[a_].is_positive:
            if (lo, hi) == (-sp.oo, sp.oo):
                return "On reconnaît l'intégrale de Gauss : $\\int_{-\\infty}^{+\\infty}e^{-a x^{2}}dx=\\sqrt{\\frac{\\pi}{a}}$ avec $a=" + sp.latex(m[a_]) + "$."
            if (lo, hi) == (0, sp.oo):
                return "On reconnaît la moitié de l'intégrale de Gauss : $\\int_{0}^{+\\infty}e^{-a x^{2}}dx=\\frac12\\sqrt{\\frac{\\pi}{a}}$ avec $a=" + sp.latex(m[a_]) + "$."
        if sp.simplify(f - sp.sin(x) / x) == 0 and (lo, hi) == (0, sp.oo):
            return "On reconnaît l'intégrale de Dirichlet, de valeur $\\frac{\\pi}{2}$ (convergence conditionnelle)."
        s_ = sp.Wild("s", exclude=[x])
        m = f.match(x ** s_ * sp.exp(-x))
        if m and (lo, hi) == (0, sp.oo) and m[s_].is_number and m[s_] > -1:
            return "On reconnaît une intégrale eulérienne : $\\int_{0}^{+\\infty}x^{s}e^{-x}dx=\\Gamma(s+1)$ avec $s=" + sp.latex(m[s_]) + "$, soit $\\Gamma(" + sp.latex(m[s_] + 1) + ")=" + sp.latex(sp.simplify(sp.gamma(m[s_] + 1))) + "$."
    except Exception:
        pass
    return None


def _exact_direct(f, x, lo, hi, sauts, ouv):
    """Valeur exacte quand aucune primitive n'est exploitable : découpage aux sauts, ou calcul direct. Renvoie (valeur, True) ou (None, False)."""
    inter = set(q for q in sauts if _num(lo) < _num(q) < _num(hi)) | _points_raccord(f, x, lo, hi)
    pts = [lo] + sorted(inter, key=lambda q: _num(q)) + [hi]
    if len(pts) > 2 and f.has(sp.Piecewise, sp.Abs):
        morceaux, tot = [], 0
        for u, v in zip(pts[:-1], pts[1:]):
            e = _expr_sur(f, x, u, v)
            val = _avec_delai(lambda: sp.integrate(e, (x, u, v)), 15)
            if val is None or val.has(sp.Integral, sp.nan, sp.zoo, sp.oo, -sp.oo):
                return None, False
            morceaux.append((u, v, e, sp.simplify(val)))
            tot += sp.simplify(val)
        bloc_m = _chaine_morceaux(f, x, lo, hi, False, False, sp.simplify(tot))
        if bloc_m:
            _P(bloc_m)
            return sp.simplify(tot), True
        desc = " ; ".join("sur $]" + sp.latex(u) + "," + sp.latex(v) + "[$, $f(x)=" + sp.latex(e) + "$" for u, v, e, _ in morceaux)
        _P(_T("Aucune primitive globale n'est utile ici : " + desc + "."))
        _P(r"\int_{" + _borne_lat(lo) + r"}^{" + _borne_lat(hi) + r"}f(x)\,dx=" + "+".join(r"\int_{" + _borne_lat(u) + r"}^{" + _borne_lat(v) + r"}" + sp.latex(e) + r"\,dx" for u, v, e, _ in morceaux)
           + "=" + "+".join(_par(val) for *_r, val in morceaux) + "=" + sp.latex(sp.simplify(tot)))
        return sp.simplify(tot), True
    val = _avec_delai(lambda: sp.integrate(f, (x, lo, hi)), 20)
    if val is None or val.has(sp.Integral, sp.nan, sp.zoo, sp.oo, -sp.oo) or val.has(*_SPECF):
        val = None
    if val is None:
        vg = _gauss_valeur_q(f, x, lo, hi)
        if vg is not None:
            blocs = _chaine_gauss(f, x, lo, hi, vg)
            if blocs:
                for b_ in blocs:
                    _P(b_)
                return vg, True
        return None, False
    val = sp.simplify(val)
    vg = _gauss_valeur_q(f, x, lo, hi)
    if vg is not None and sp.simplify(vg - val) == 0:
        blocs = _chaine_gauss(f, x, lo, hi, vg)
        if blocs:
            for b_ in blocs:
                _P(b_)
            return val, True
    vn = _quad(_fnum(f, x), lo, hi) if _fnum(f, x) is not None else None
    if vn is None or _num(val) is None or abs(_num(val) - vn) > 1e-6 * (1 + abs(vn)):
        return None, False
    _P(_T("Sur $" + ouv + "$, $f$ n'a pas de primitive qui s'exprime avec les fonctions usuelles ; l'intégrale se calcule néanmoins : $\\int_{" + _borne_lat(lo) + "}^{" + _borne_lat(hi) + "}f(x)\\,dx=" + sp.latex(val)
          + "$ (valeur contrôlée numériquement)."))
    return val, True


def _etude_morceau(f, x, fm, m, k, nb, absv, sauts=()):
    """Écrit le calcul sur le morceau m ; absv[(c,p)] = verdict d'intégrabilité de chaque demi-intégrale."""
    lo, hi = m["lo"], m["hi"]
    pl, ph = m["pb_lo"], m["pb_hi"]
    res = dict(lo=lo, hi=hi, pl=pl, ph=ph, valeur=None, conv=None, F=None, halves=[], num=None, type=None)
    nbpb = int(pl) + int(ph)
    res["type"] = "propre" if nbpb == 0 else ("impropre" if nbpb == 1 else "généralisée")
    ouv = _iv(lo, hi, pl, ph)
    kn = _connue(f, x, lo, hi)
    F = _primitive(f, x, lo, hi)
    if F is not None:
        for (e_, s_) in ((lo, "+"), (hi, "-")):
            cl_t, _v = _valeur_F(F, x, e_, s_)
            if cl_t not in ("fin", "inf", "osc"):
                F = None
                break
    res["F"] = F
    if F is None and kn:
        _P(_T(kn))
    if F is not None:
        # chaînes d'égalités alignées (une par demi-intégrale) ; repli sur la rédaction en phrases si l'une échoue
        chaines = []
        if pl and ph:
            c0 = _point_coupure(lo, hi)
            parts0 = [("lo", lo, c0), ("hi", c0, hi)]
        else:
            parts0 = [("all", lo, hi)]
        for tag0, u0, v0 in parts0:
            pb_u0 = (u0 in (sp.oo, -sp.oo)) or (tag0 in ("lo", "all") and pl)
            pb_v0 = (v0 in (sp.oo, -sp.oo)) or (tag0 in ("hi", "all") and ph)
            cu0, Fu0 = _valeur_F(F, x, u0, "+")
            cv0, Fv0 = _valeur_F(F, x, v0, "-")
            bloc0 = None
            if cu0 == "fin" and cv0 == "fin":
                val0 = sp.simplify(Fv0 - Fu0)
                bloc0 = _chaine_int(f, x, u0, v0, pb_u0, pb_v0, val0)
            elif "inf" in (cu0, cv0) and "osc" not in (cu0, cv0) and "inc" not in (cu0, cv0):
                try:
                    sg0 = sp.oo if (cv0 == "inf" and Fv0 == sp.oo) or (cu0 == "inf" and Fu0 == -sp.oo) else -sp.oo
                    bloc0 = _chaine_int(f, x, u0, v0, pb_u0, pb_v0, sg0)
                except Exception:
                    bloc0 = None
            chaines.append(bloc0)
        chaine_ok = all(b_ is not None for b_ in chaines)
        if kn and not chaine_ok:
            _P(_T(kn))
        lg_n = [] if chaine_ok else _narrer_primitive(f, x, F)
        if chaine_ok:
            pass
        elif lg_n:
            for l_ in lg_n:
                _P(l_)
        else:
            _P(_T("Sur $" + ouv + "$, une primitive de $f$ est $F(x)=" + sp.latex(F) + "$."))
            met = _methode_texte(f, x)
            if met:
                _P(met)
        if chaine_ok:
            pass
        elif pl or ph:
            if _premiere("tfa"):
                _P(_T("Comme $f$ est continue sur " + "$" + ouv + "$" + ", le théorème fondamental de l'analyse donne $\\int_{u}^{X}f=F(X)-F(u)$ sur tout segment $[u,X]$ de cet intervalle ; l'intégrale impropre s'obtient en faisant tendre $X$ vers la borne à problème."))
        elif _premiere("tfa"):
            _P(_T("Comme $f$ est continue sur le segment $[" + sp.latex(lo) + "," + sp.latex(hi) + "]$, le théorème fondamental de l'analyse donne $\\int_{" + sp.latex(lo) + "}^{" + sp.latex(hi) + "}f=F(" + sp.latex(hi) + ")-F(" + sp.latex(lo) + ")$."))
        c = None
        if pl and ph:
            c = _point_coupure(lo, hi)
            parts = [("lo", lo, c), ("hi", c, hi)]
        else:
            parts = [("all", lo, hi)]
        total = 0
        tout_fin = True
        for idx_p, (tag, u, v) in enumerate(parts):
            pb_u = (u in (sp.oo, -sp.oo)) or (tag in ("lo", "all") and pl)
            pb_v = (v in (sp.oo, -sp.oo)) or (tag in ("hi", "all") and ph)
            cl_u, Fu = _valeur_F(F, x, u, "+")
            cl_v, Fv = _valeur_F(F, x, v, "-")
            faits = []
            for (cl_, F_, pt_, sens_, pb_) in ((cl_u, Fu, u, "+", pb_u), (cl_v, Fv, v, "-", pb_v)):
                faits.append((pb_, _fait_F(cl_, F_, pt_, sens_, pb_, F, x)))
            faits.sort(key=lambda t: 0 if t[0] else 1)
            core = faits[0][1] + ((" et " + faits[1][1]) if len(faits) > 1 else "")
            half = dict(u=u, v=v, pb_u=pb_u, pb_v=pb_v, conv=None, valeur=None)
            integ = r"\int_{" + _borne_lat(u) + r"}^{" + _borne_lat(v) + r"}f(x)\,dx"
            if cl_u == "fin" and cl_v == "fin":
                val = sp.simplify(Fv - Fu)
                half.update(conv=True, valeur=val)
                egal = "$" + integ + "=" + _par(Fv) + "-" + _par(Fu) + "=" + sp.latex(val) + "$"
                if chaine_ok:
                    _P(chaines[idx_p])
                else:
                    _P(_T(_V("calc", "On a " + core + ", d'où " + egal + ".", "Comme " + core + ", il vient " + egal + ".", "Ici " + core + " ; par conséquent " + egal + ".")))
                total += val
            else:
                tout_fin = False
                half.update(conv=False)
                if "inf" in (cl_u, cl_v) and "osc" not in (cl_u, cl_v) and "inc" not in (cl_u, cl_v):
                    try:
                        sig = r"+\infty" if (cl_v == "inf" and Fv == sp.oo) or (cl_u == "inf" and Fu == -sp.oo) else r"-\infty"
                    except Exception:
                        sig = r"\pm\infty"
                    if chaine_ok:
                        _P(chaines[idx_p])
                    else:
                        _P(_T(_V("calcd", "On a " + core + ", donc $" + integ + "=" + sig + "$ : divergente.", "Comme " + core + ", $" + integ + "=" + sig + "$ et l'intégrale diverge.")))
                    half["valeur"] = sig
                elif "osc" in (cl_u, cl_v):
                    _P(_T("Ici " + core + " : l'intégrale $" + integ + "$ oscille sans se stabiliser et diverge."))
                else:
                    _P(_T("On a " + core + "."))
                    half["conv"] = None
            res["halves"].append(half)
        if tout_fin:
            res["valeur"] = sp.simplify(total)
            res["conv"] = True
            if len(parts) == 2:
                _P(_T("Par somme, $\\int_{" + _borne_lat(lo) + "}^{" + _borne_lat(hi) + "}f(x)\\,dx=" + " + ".join(_par(h["valeur"]) for h in res["halves"]) + "=" + sp.latex(res["valeur"]) + "$."))
        else:
            res["conv"] = False if all(h["conv"] is False for h in res["halves"] if h["conv"] is not None) and any(h["conv"] is False for h in res["halves"]) else None
    else:
        c = _point_coupure(lo, hi) if (pl and ph) else None
        todo = []
        if pl:
            todo.append((c if c is not None else hi, lo))
        if ph:
            todo.append((c if c is not None else lo, hi))
        for (cc, pp) in todo:
            res["halves"].append(dict(u=cc, v=pp, pb_u=False, pb_v=True, conv=absv.get((cc, pp)), valeur=None, c=cc, p=pp))
        res["conv"] = True if (not pl and not ph) else (True if res["halves"] and all(h["conv"] is True for h in res["halves"]) else None)
        val_ex, ok_ex = (None, False)
        if res["conv"]:
            val_ex, ok_ex = _exact_direct(f, x, lo, hi, sauts, ouv)
        if ok_ex:
            res["valeur"] = val_ex
        else:
            vn = _quad(fm, lo, hi)
            if vn is not None and res["conv"]:
                res["num"] = vn
                _P(_T("Sur $" + ouv + "$, aucune primitive de $f$ n'est exprimable avec les fonctions usuelles : la valeur ne peut pas se lire par $F(b)-F(a)$. L'intégrabilité étant acquise, on la calcule numériquement : $\\int_{"
                      + _borne_lat(lo) + "}^{" + _borne_lat(hi) + "}f(x)\\,dx\\approx " + sp.latex(sp.Float(vn, 8)) + "$."))
            else:
                _P(_T("Sur $" + ouv + "$, aucune primitive de $f$ n'est exprimable avec les fonctions usuelles."))
    return res




# -------------------------------------------------------------
# Valeurs numériques d'appui
# -------------------------------------------------------------
def _fnp(f, x):
    try:
        g = sp.lambdify(x, f, modules=["numpy"])

        def h(t):
            with np.errstate(all="ignore"):
                r = g(t)
            r = np.asarray(r, dtype=float)
            if r.shape != np.shape(t):
                r = np.broadcast_to(r, np.shape(t)).copy()
            return r
        h(np.array([0.5]))
        return h
    except Exception:
        return None


def _riemann(fn, lo, hi, n):
    """(gauche, droite, milieu) pour n sous-intervalles."""
    h = (hi - lo) / n
    xs = lo + h * np.arange(n + 1)
    mid = lo + h * (np.arange(n) + 0.5)
    yl, yr, ym = fn(xs[:-1]), fn(xs[1:]), fn(mid)
    return float(np.sum(yl) * h), float(np.sum(yr) * h), float(np.sum(ym) * h)


def _etagee(fn, lo, hi, N, M=400001):
    """Intégrales de s_N^+ et s_N^- (fonctions étagées de Lebesgue) et de f^+, f^-, sur [lo, hi]."""
    t = lo + (hi - lo) * (np.arange(M) + 0.5) / M
    y = fn(t)
    y = np.where(np.isfinite(y), y, 0.0)
    ypos, yneg = np.maximum(y, 0), np.maximum(-y, 0)
    sp_ = np.minimum(np.floor(ypos * 2 ** N) / 2 ** N, N)
    sn_ = np.minimum(np.floor(yneg * 2 ** N) / 2 ** N, N)
    L = hi - lo
    return float(sp_.mean() * L), float(sn_.mean() * L), float(ypos.mean() * L), float(yneg.mean() * L)


# -------------------------------------------------------------
# Étude complète
# -------------------------------------------------------------
def _decrit_lim(q, sens, g):
    """Phrase sur la limite de f en q (côté sens), g = (classe, valeur) ou None."""
    quand = "quand $x\\to " + sp.latex(q) + ("^{-}" if sens == "-" else "^{+}") + "$"
    if g is None:
        return None
    if g[0] == "fin":
        return "$f(x)\\to " + sp.latex(g[1]) + "$ " + quand
    if g[0] == "inf":
        return "$f(x)\\to " + _lat(g[1]) + "$ " + quand
    return "$f$ n'a pas de limite " + quand


def _demi_integrales(morceaux):
    """Liste des (c, p) : une borne à problème p, un point régulier c."""
    out = []
    for m in morceaux:
        lo, hi, pl, ph = m["lo"], m["hi"], m["pb_lo"], m["pb_hi"]
        if pl and ph:
            c = _point_coupure(lo, hi)
            out += [(c, lo), (c, hi)]
        elif pl:
            out.append((hi, lo))
        elif ph:
            out.append((lo, hi))
    return out


def etude_integrale(f_expr, x, intervalle, N=10):
    _VUES.clear()
    _GAUSS_Q.clear()
    _ROT.clear()
    _REFS.clear()
    P = _P
    try:
        f = sp.sympify(f_expr)
    except Exception:
        _rouge("Expression invalide pour f.")
        return
    if intervalle == S.Reals or not isinstance(intervalle, sp.Interval):
        a, b, oa, ob = -sp.oo, sp.oo, True, True
    else:
        a, b, oa, ob = intervalle.start, intervalle.end, intervalle.left_open, intervalle.right_open
    if a == b or (_num(a) is not None and _num(b) is not None and _num(a) >= _num(b)):
        _rouge("L'intervalle est vide ou réduit à un point : l'intégrale vaut 0.")
        return

    _titre("Données")
    P(r"\int_{" + _borne_lat(a) + r"}^{" + _borne_lat(b) + r"} f(x)\,dx\quad\text{avec}\quad f(x)=" + sp.latex(f) + r"\quad\text{sur } I=" + _iv(a, b, oa, ob))

    morceaux, infos = _decoupage(f, x, a, b)
    if morceaux is None:
        _rouge(infos)
        return
    fm = _fnum(f, x)
    fn = _fnp(f, x)
    halves_all = _demi_integrales(morceaux)
    nbpb_tot = sum(int(m["pb_lo"]) + int(m["pb_hi"]) for m in morceaux)

    # ---- 1) points à problème et découpage -----------------------------------------
    _titre("Points à problème et découpage")
    bornes_inf = [e for e in (a, b) if e in (sp.oo, -sp.oo)]
    if infos["cand"] or bornes_inf:
        for q in infos["cand"]:
            g, d, ok = infos["diag"][q]
            ph_ = [t for t in (_decrit_lim(q, "-", g), _decrit_lim(q, "+", d)) if t]
            if ok:
                if g is not None and d is not None and g[0] == "fin" and d[0] == "fin" and sp.simplify(g[1] - d[1]) != 0:
                    verdict = " : $f$ présente un saut fini, elle reste bornée au voisinage de $" + sp.latex(q) + "$ et ce point ne pose pas de problème."
                elif g is not None and d is not None and g[0] == "fin" and d[0] == "fin":
                    verdict = " : les deux limites sont égales, donc $f$ se prolonge par continuité en $" + sp.latex(q) + "$ et ce point ne pose pas de problème."
                else:
                    verdict = " : cette limite est finie, donc $f$ se prolonge par continuité en $" + sp.latex(q) + "$ et ce point ne pose pas de problème."
            else:
                verdict = (" : donc $f$ n'est pas bornée au voisinage de $" if any(t is not None and t[0] == "inf" for t in (g, d)) else " : $f$ n'a pas de limite finie, elle n'est pas contrôlée au voisinage de $") + sp.latex(q) + "$ ; c'est un point à problème."
            P(_T("En $x=" + sp.latex(q) + "$, " + " et ".join(ph_) + verdict))
        if bornes_inf:
            if len(bornes_inf) == 2:
                P(_T("Les bornes $-\\infty$ et $+\\infty$ sont infinies : ce sont aussi des points à problème."))
            else:
                P(_T("La borne $" + _borne_lat(bornes_inf[0]) + "$ est infinie : c'est aussi un point à problème."))
    else:
        P(_T("La fonction $f$ est continue sur le segment $[" + sp.latex(a) + "," + sp.latex(b) + "]$ : aucun point ne pose problème."))
    if nbpb_tot == 0:
        if infos["cand"]:
            P(_T("Donc $f$ est continue par morceaux sur $[" + sp.latex(a) + "," + sp.latex(b) + "]$ avec des limites finies de part et d'autre de chaque discontinuité : l'intégrale est propre."))
        else:
            P(_T("L'intégrale est donc propre."))
    else:
        if nbpb_tot == 1:
            P(_T("Il n'y a qu'un seul point à problème : l'intégrale est impropre, et aucun découpage n'est nécessaire."))
        else:
            ints = infos["interieurs"]
            cs = [c for (c, p) in halves_all if any(m["pb_lo"] and m["pb_hi"] and m["lo"] == p for m in morceaux)]
            cs = list(dict.fromkeys(cs))
            motifs = []
            if ints:
                motifs.append("aux points à problème intérieurs " + ", ".join("$x=" + sp.latex(q) + "$" for q in ints))
            if cs:
                motifs.append("en " + ", ".join("$c=" + sp.latex(c) + "$" for c in cs) + " (point régulier choisi pour séparer deux bornes à problème)")
            P(_T("Pour que chaque intégrale n'ait plus qu'un seul point à problème, on découpe " + " puis ".join(motifs) + " :"))
            termes = []
            for m in morceaux:
                lo, hi, pl, ph = m["lo"], m["hi"], m["pb_lo"], m["pb_hi"]
                if pl and ph:
                    c = _point_coupure(lo, hi)
                    termes += [r"\int_{" + _borne_lat(lo) + r"}^{" + sp.latex(c) + r"}f", r"\int_{" + sp.latex(c) + r"}^{" + _borne_lat(hi) + r"}f"]
                else:
                    termes.append(r"\int_{" + _borne_lat(lo) + r"}^{" + _borne_lat(hi) + r"}f")
            P(r"\int_{" + _borne_lat(a) + r"}^{" + _borne_lat(b) + r"}f(x)\,dx=" + " + ".join(termes))
            P(_T("L'intégrale est donc généralisée ; chacune de ces intégrales est étudiée séparément."))

    # ---- 2) intégrabilité -------------------------------------------------------------------
    _titre("Intégrabilité de f sur I")
    if nbpb_tot > 0:
        P(_T("Au sens de Lebesgue, $f$ est intégrable sur $I$ ($f\\in L^1(I)$) lorsque $f$ est mesurable et $\\int_I|f|\\,d\\lambda<+\\infty$. Pour une fonction continue par morceaux sur un intervalle, cela signifie que l'intégrale impropre de $|f|$ converge : c'est la convergence absolue."))
        P(_T("On étudie donc d'abord $|f|$ (positive, donc sans compensation possible) ; l'intégrale impropre de $f$ elle-même est traitée ensuite."))
    absv = {}
    l1_parts = []
    for k, m in enumerate(morceaux, 1):
        if not m["pb_lo"] and not m["pb_hi"]:
            seg = "[" + sp.latex(m["lo"]) + "," + sp.latex(m["hi"]) + "]"
            if not infos["cand"]:
                P(_T("Sur $" + seg + "$, $f$ est continue sur un segment, donc bornée (théorème des bornes atteintes) ; ce segment étant de mesure finie, $f$ y est intégrable."))
            else:
                P(_T("Sur $" + seg + "$, $f$ est continue par morceaux avec des limites finies de part et d'autre de chaque discontinuité : elle est bornée sur un segment de mesure finie, donc intégrable."))
            l1_parts.append(True)
    for (c, p) in halves_all:
        v_abs, lg = _abs_integrable(f, x, fm, c, p)
        for l in lg:
            P(l)
        absv[(c, p)] = v_abs
        l1_parts.append(v_abs)
        if v_abs is None:
            P(_T("On ne conclut pas sur $" + _interv_demi(c, p) + "$."))
    l1 = len(l1_parts) > 0 and all(v is True for v in l1_parts)
    if l1:
        P(_T("Toutes les parties sont intégrables, donc $\\int_I|f|\\,d\\lambda<+\\infty$ : $f\\in L^1(I)$."))
    elif any(v is False for v in l1_parts):
        P(_T("Une partie au moins n'est pas intégrable, donc $\\int_I|f|\\,d\\lambda=+\\infty$ : $f\\notin L^1(I)$."))

    # ---- 3) calcul ----------------------------------------------------------------------------
    _titre("Calcul de l'intégrale")
    if not l1:
        P(_T("L'intégrale impropre se définit comme la limite des intégrales sur des segments, $\\int_a^b f=\\lim_{X\\to b}\\int_a^X f$ : aucune valeur absolue n'y intervient. Elle peut donc converger sans que $f$ soit intégrable, lorsque les parties positive et négative de $f$, d'intégrales infinies chacune, se compensent dans la limite. On la calcule directement."))
    elif nbpb_tot > 0:
        P(_T("Comme $f\\in L^1(I)$, l'intégrale impropre converge et coïncide avec l'intégrale de Lebesgue : c'est le théorème de convergence dominée appliqué à $f\\,\\mathbf 1_{[a,X]}$, dominée par $|f|$."))
    resultats = []
    for k, m in enumerate(morceaux, 1):
        resultats.append(_etude_morceau(f, x, fm, m, k, len(morceaux), absv, [q for q in infos["cand"] if q not in infos["interieurs"] and q not in (a, b)]))

    # ---- 4) bilan -------------------------------------------------------------------------------
    _titre("Bilan")
    conv_all = all(r["conv"] is True for r in resultats)
    valeur_tot = None
    if conv_all and all(r["valeur"] is not None for r in resultats):
        valeur_tot = sp.simplify(sum(r["valeur"] for r in resultats))
    num_tot = None
    if valeur_tot is not None:
        num_tot = _num(valeur_tot)
    elif conv_all and all((r["valeur"] is not None or r["num"] is not None) for r in resultats):
        num_tot = sum((_num(r["valeur"]) if r["valeur"] is not None else r["num"]) for r in resultats)
    integrale = r"\int_{" + _borne_lat(a) + r"}^{" + _borne_lat(b) + r"}f(x)\,dx"
    if conv_all and l1:
        if valeur_tot is not None:
            P(_T("Ainsi $f\\in L^1(I)$ et $\\int_I f\\,d\\lambda=" + integrale + "=" + sp.latex(valeur_tot) + "$."))
        elif num_tot is not None:
            P(_T("Ainsi $f\\in L^1(I)$ et $\\int_I f\\,d\\lambda=" + integrale + "\\approx " + sp.latex(sp.Float(num_tot, 8)) + "$ (valeur numérique)."))
        if fm is not None and num_tot is not None:
            ctrl, ok = 0.0, True
            for r in resultats:
                q = _quad(fm, r["lo"], r["hi"])
                if q is None:
                    ok = False
                    break
                ctrl += q
            if ok:
                ecart = abs(ctrl - num_tot)
                if ecart <= 1e-5 * (1 + abs(num_tot)):
                    P(_T("Contrôle numérique : $" + sp.latex(sp.Float(ctrl, 8)) + "$ (écart $" + sp.latex(sp.Float(ecart, 2)) + "$)."))
                elif nbpb_tot > 0:
                    P(_T("La quadrature numérique directe donne $" + sp.latex(sp.Float(ctrl, 6)) + "$ : la queue décroît trop lentement pour être captée numériquement ; la valeur exacte vient de l'étude ci-dessus."))
    elif conv_all and not l1:
        if valeur_tot is not None:
            P(_T("L'intégrale impropre converge et vaut $" + sp.latex(valeur_tot) + "$ (convergence conditionnelle)."))
        P(_T("Pourtant $\\int_I|f|=+\\infty$ : $f\\notin L^1(I)$ et son intégrale de Lebesgue n'est pas définie. En effet $\\int f^+$ et $\\int f^-$ valent chacune $+\\infty$ : les aires des arches positives et négatives forment des séries divergentes."))
        P(_T("L'intégrale impropre converge néanmoins parce que ces arches alternent et que leurs aires décroissent vers $0$ : les sommes partielles $\\int_a^X f$ s'annulent presque, exactement comme les sommes d'une série alternée convergente sans convergence absolue (convergence conditionnelle)."))
    else:
        P(_T("L'intégrale diverge (au moins une des intégrales partielles ne converge pas) : $f\\notin L^1(I)$."))
        try:
            _valeur_principale(f, x, resultats, a, b)
        except Exception:
            pass

    # ---- tables numériques -----------------------------------------------------------------
    tous_propres = all(r["type"] == "propre" for r in resultats)
    if tous_propres and fn is not None and conv_all and num_tot is not None:
        _titre("Illustration numérique : sommes de Riemann et approximation de Lebesgue")
        lo_f, hi_f = float(sp.N(a)), float(sp.N(b))
        V = num_tot
        lignes_n = {}
        for n_ in (10, 100, 1000):
            try:
                lignes_n[n_] = _riemann(fn, lo_f, hi_f, n_)
            except Exception:
                lignes_n = {}
                break
        if lignes_n:
            P(r"\text{Sommes de Riemann (pas } h=\frac{b-a}{n}\text{) à comparer à } V=" + sp.latex(sp.Float(V, 8)) + r"\text{ :}")
            for n_, (g_, d_, m_) in lignes_n.items():
                P(r"n=" + str(n_) + r"\ :\ \underbrace{" + sp.latex(sp.Float(g_, 7)) + r"}_{\text{gauche}},\ \underbrace{" + sp.latex(sp.Float(d_, 7)) + r"}_{\text{droite}},\ \underbrace{" + sp.latex(sp.Float(m_, 7)) + r"}_{\text{milieu}}")
            e10 = [abs(v - V) for v in lignes_n[10]]
            e100 = [abs(v - V) for v in lignes_n[100]]
            P(r"\text{Erreurs } |S_n-V| \text{ pour } n=10 \to n=100 \text{ : gauche } " + sp.latex(sp.Float(e10[0], 3)) + r"\to " + sp.latex(sp.Float(e100[0], 3)) + r",\ \text{droite } " + sp.latex(sp.Float(e10[1], 3)) + r"\to " + sp.latex(sp.Float(e100[1], 3))
              + r",\ \text{milieu } " + sp.latex(sp.Float(e10[2], 3)) + r"\to " + sp.latex(sp.Float(e100[2], 3)))
            try:
                rg = e10[0] / e100[0]
                rm = e10[2] / e100[2]
                P(r"\text{Rapport d'erreur (n multiplié par 10) : gauche/droite } \approx " + sp.latex(sp.Float(rg, 3)) + r"\text{ (erreur en } 1/n\text{), milieu } \approx " + sp.latex(sp.Float(rm, 3)) + r"\text{ (erreur en } 1/n^2\text{ si } f \in C^2\text{).}")
            except Exception:
                pass
        try:
            P(r"\text{Approximation de Lebesgue : } s_N^{\pm}=\min\!\left(N,\frac{\lfloor 2^N f^{\pm}\rfloor}{2^N}\right) \text{ (fonction étagée, constante sur les ensembles } \{k2^{-N}\le f^{\pm}<(k+1)2^{-N}\}\text{) :}")
            for N_ in (1, 2, 4, 8):
                ip, in_, fp, fneg = _etagee(fn, lo_f, hi_f, N_)
                P(r"N=" + str(N_) + r"\ :\ \int s_N^+=" + sp.latex(sp.Float(ip, 6)) + r",\ \int s_N^-=" + sp.latex(sp.Float(in_, 6)) + r"\ \Rightarrow\ \int s_N^+-\int s_N^-=" + sp.latex(sp.Float(ip - in_, 7)))
            P(r"\text{Ces valeurs croissent vers } \int f^+=" + sp.latex(sp.Float(fp, 6)) + r"\text{ et } \int f^-=" + sp.latex(sp.Float(fneg, 6)) + r"\text{ (convergence monotone), et leur différence tend vers } V=" + sp.latex(sp.Float(V, 7)) + r".")
        except Exception:
            pass
    elif conv_all:
        _titre("Illustration numérique : intégrales tronquées")
        for r in resultats:
            for h in r["halves"]:
                if not (h.get("pb_u") or h.get("pb_v")):
                    continue
                if "c" in h:
                    cc, pp = h["c"], h["p"]
                else:
                    cc, pp = (h["u"], h["v"]) if h.get("pb_v") else (h["v"], h["u"])
                if fm is None:
                    continue
                if pp in (sp.oo, -sp.oo):
                    sg = 1 if pp == sp.oo else -1
                    cf = float(sp.N(cc))
                    Xs = [cf + sg * 10.0 ** k for k in (1, 2, 3)]
                else:
                    pf, cf = float(sp.N(pp)), float(sp.N(cc))
                    Xs = [pf + (cf - pf) * 10.0 ** (-k) for k in (1, 2, 3)]
                vals = []
                for Xv in Xs:
                    lo_, hi_ = (cf, Xv) if Xv > cf else (Xv, cf)
                    q = _avec_delai(lambda: float(mp.quad(lambda t: fm(t), mp.linspace(mp.mpf(lo_), mp.mpf(hi_), 41))), 20)
                    vals.append(q)
                if all(v is not None for v in vals):
                    cible = ""
                    if h.get("valeur") is not None and h["valeur"] not in (sp.oo, -sp.oo):
                        try:
                            cible = r"\ \longrightarrow\ " + sp.latex(sp.Float(float(sp.N(h["valeur"])), 7)) + r"\ (\text{valeur de cette intégrale})"
                        except Exception:
                            cible = ""
                    sens_bas = Xs[0] < cf
                    cf_l = sp.latex(sp.Rational(cf).limit_denominator(1000)) if abs(cf - float(sp.Rational(cf).limit_denominator(1000))) < 1e-12 else sp.latex(sp.Float(cf, 4))
                    borne_txt = (r"\int_{X}^{" + cf_l + r"}f") if sens_bas else (r"\int_{" + cf_l + r"}^{X}f")
                    P(borne_txt + r" \text{ pour } X=" + ",\\ ".join(sp.latex(sp.Float(v_, 6)) for v_ in Xs) + r"\text{ : } " + ",\\ ".join(sp.latex(sp.Float(v_, 7)) for v_ in vals) + cible)

    # ---- graphes -----------------------------------------------------------------------------
    try:
        _graphes_integrale(f, x, fn, fm, a, b, resultats, conv_all, num_tot, l1)
    except Exception as e:  # pragma: no cover
        _rouge("Les graphes n'ont pas pu être produits : " + str(e))


def _valeur_principale(f, x, resultats, a, b):
    """Valeur principale de Cauchy si elle existe (divergence par compensation symétrique)."""
    pts_int = [r["hi"] for r in resultats[:-1]]
    eps = sp.Symbol("epsilon", positive=True)
    if len(resultats) == 1 and a == -sp.oo and b == sp.oo and resultats[0]["F"] is not None:
        F = resultats[0]["F"]
        X = sp.Symbol("X", positive=True)
        vp = _avec_delai(lambda: sp.limit(F.subs(x, X) - F.subs(x, -X), X, sp.oo), 15)
        cl, v = _classe_limite(vp)
        if cl == "fin":
            st_ = _T("Valeur principale de Cauchy :") + r"\ \lim_{X\to+\infty}\int_{-X}^{X}f=\lim_{X\to+\infty}\big(F(X)-F(-X)\big)=" + sp.latex(v)
            _P(st_ + _T(" (ce n'est pas la valeur de l'intégrale, qui diverge : la compensation n'a lieu que pour des bornes symétriques)."))
    elif len(resultats) == 2 and resultats[0]["F"] is not None and resultats[1]["F"] is not None:
        p = resultats[0]["hi"]
        F1, F2 = resultats[0]["F"], resultats[1]["F"]
        lo_c = _valeur_F(F1, x, a, "+")
        hi_c = _valeur_F(F2, x, b, "-")
        if lo_c[0] == "fin" and hi_c[0] == "fin":
            expr = F1.subs(x, p - eps) - F2.subs(x, p + eps)
            vp = _avec_delai(lambda: sp.limit(expr, eps, 0, "+"), 15)
            cl, v = _classe_limite(vp)
            if cl == "fin":
                tot = sp.simplify(v - lo_c[1] + hi_c[1])
                def _pm(sg):
                    return (sp.latex(p) + sg + r"\varepsilon") if p != 0 else ((r"" if sg == "+" else sg) + r"\varepsilon")
                _P(_T("Valeur principale de Cauchy en $x=" + sp.latex(p) + "$ :") + r"\ \lim_{\varepsilon\to0^+}\left(\int_{" + _borne_lat(a) + r"}^{" + _pm("-") + r"}f+\int_{" + _pm("+") + r"}^{" + _borne_lat(b) + r"}f\right)="
                   + sp.latex(tot) + _T(" (les deux divergences se compensent ; ce n'est pas la valeur de l'intégrale, qui diverge)."))




# -------------------------------------------------------------
# Graphes
# -------------------------------------------------------------
def _style_axes(ax, titre=None, xl=None, yl=None):
    ax.grid(True, alpha=0.25)
    if titre:
        ax.set_title(titre, fontsize=11)
    if xl:
        ax.set_xlabel(xl)
    if yl:
        ax.set_ylabel(yl)


def _fenetre(lo, hi, R=12.0):
    """Fenêtre d'affichage finie pour un morceau (bornes infinies remplacées)."""
    lo_f = None if lo == -sp.oo else float(sp.N(lo))
    hi_f = None if hi == sp.oo else float(sp.N(hi))
    if lo_f is None and hi_f is None:
        return -R, R
    if lo_f is None:
        return hi_f - R, hi_f
    if hi_f is None:
        return lo_f, lo_f + R
    return lo_f, hi_f


def _graphes_integrale(f, x, fn, fm, a, b, resultats, conv_all, num_tot, l1):
    if fn is None:
        return
    _titre("Graphes")
    # ---------- 1) la fonction et l'aire --------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ymax_all = []
    for r in resultats:
        wl, wh = _fenetre(r["lo"], r["hi"])
        L = wh - wl
        eps = L * 1e-7
        xs = np.linspace(wl + eps, wh - eps, 6001)
        ys = fn(xs)
        ys = np.where(np.isfinite(ys), ys, np.nan)
        fin = ys[np.isfinite(ys)]
        if fin.size:
            ymax_all.append(np.percentile(np.abs(fin), 97))
        couleur = "tab:blue" if conv_all else "tab:orange"
        ax.plot(xs, ys, color="tab:blue", lw=1.6)
        ax.fill_between(xs, 0, np.where(np.isfinite(ys), ys, 0), alpha=0.25, color=couleur)
        for e, pb in ((r["lo"], r["pl"]), (r["hi"], r["ph"])):
            if pb and e not in (sp.oo, -sp.oo):
                ax.axvline(float(sp.N(e)), color="crimson", ls="--", lw=1.2)
    if ymax_all:
        Y = max(ymax_all) * 1.6
        Y = max(Y, 1e-9)
        ax.set_ylim(-Y if any(np.nanmin(fn(np.linspace(*_fenetre(r['lo'], r['hi']), 400)[1:-1])) < 0 for r in resultats) else -0.1 * Y, Y)
    ax.axhline(0, color="k", lw=0.8)
    inf_txt = " (fenêtre limitée à ±12 : la borne infinie est tronquée à l'affichage)" if any(r["lo"] == -sp.oo or r["hi"] == sp.oo for r in resultats) else ""
    _style_axes(ax, "f et aire algébrique" + inf_txt, "x", "f(x)")
    st.pyplot(fig)

    tous_propres = all(r["type"] == "propre" for r in resultats)
    # ---------- 2) Riemann ------------------------------------------------------------
    if tous_propres and num_tot is not None and conv_all:
        lo_f, hi_f = float(sp.N(a)), float(sp.N(b))
        fig, axs = plt.subplots(2, 2, figsize=(11, 7.4))
        xs = np.linspace(lo_f, hi_f, 1500)
        ys = fn(xs)
        n0 = 8
        h = (hi_f - lo_f) / n0
        g = lo_f + h * np.arange(n0)
        regles = [("gauche", g), ("milieu", g + h / 2), ("droite", g + h)]
        for ax, (nom, pts) in zip((axs[0, 0], axs[0, 1], axs[1, 0]), regles):
            ax.plot(xs, ys, color="tab:blue", lw=1.6)
            hts = fn(pts)
            ax.bar(g, hts, width=h, align="edge", alpha=0.35, color="tab:orange", edgecolor="k")
            ax.axhline(0, color="k", lw=0.6)
            _style_axes(ax, "Somme de Riemann (" + nom + "), n=8 : " + ("%.5g" % float(np.sum(hts) * h)), "x")
        ns = np.unique(np.round(np.logspace(1, 3.3, 14)).astype(int))
        err = {"gauche": [], "milieu": [], "droite": []}
        for n_ in ns:
            gl, dr, mi = _riemann(fn, lo_f, hi_f, int(n_))
            err["gauche"].append(abs(gl - num_tot)); err["milieu"].append(abs(mi - num_tot)); err["droite"].append(abs(dr - num_tot))
        ax = axs[1, 1]
        for nom, col in (("gauche", "tab:green"), ("milieu", "tab:red"), ("droite", "tab:purple")):
            ax.loglog(ns, np.maximum(err[nom], 1e-17), "o-", ms=3, label=nom, color=col)
        ax.loglog(ns, err["gauche"][0] * ns[0] / ns, "k--", lw=0.8, label="1/n")
        ax.loglog(ns, max(err["milieu"][0], 1e-17) * (ns[0] / ns) ** 2, "k:", lw=0.8, label="1/n²")
        ax.legend(fontsize=8)
        _style_axes(ax, "Erreur |S_n − V| en fonction de n", "n", "erreur")
        fig.tight_layout()
        st.pyplot(fig)

        # ---------- 3) Lebesgue : fonctions étagées et fonction de répartition -------
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
        ax = axs[0]
        M = 6000
        tt = lo_f + (hi_f - lo_f) * (np.arange(M) + 0.5) / M
        yy = fn(tt)
        ax.plot(tt, yy, color="tab:blue", lw=1.8, label="f")
        for N_, col in ((1, "tab:green"), (2, "tab:orange"), (3, "tab:red")):
            sp_ = np.minimum(np.floor(np.maximum(yy, 0) * 2 ** N_) / 2 ** N_, N_)
            sn_ = np.minimum(np.floor(np.maximum(-yy, 0) * 2 ** N_) / 2 ** N_, N_)
            ax.plot(tt, sp_ - sn_, color=col, lw=1.0, label="s_%d⁺ − s_%d⁻ (étagée)" % (N_, N_))
        ax.axhline(0, color="k", lw=0.6)
        ax.legend(fontsize=8)
        _style_axes(ax, "Approximation de Lebesgue par des fonctions étagées", "x")
        ax = axs[1]
        ymax = float(np.max(yy)); ymin = float(np.min(yy))
        L = hi_f - lo_f
        if ymax > 0:
            ts = np.linspace(0, ymax, 400)
            mplus = np.array([np.mean(yy > t) * L for t in ts])
            ax.plot(ts, mplus, color="tab:blue", label="λ({f > t})")
            ax.fill_between(ts, 0, mplus, alpha=0.25, color="tab:blue")
        if ymin < 0:
            ts2 = np.linspace(0, -ymin, 400)
            mmoins = np.array([np.mean(yy < -t) * L for t in ts2])
            ax.plot(ts2, mmoins, color="tab:red", label="λ({f < −t})")
            ax.fill_between(ts2, 0, mmoins, alpha=0.2, color="tab:red")
        ax.legend(fontsize=8)
        _style_axes(ax, "Fonction de répartition t ↦ λ({f>t}) : aire = ∫f⁺ (formule de Cavalieri)", "t", "mesure")
        fig.tight_layout()
        st.pyplot(fig)

    # ---------- 4) intégrale cumulée vers chaque bord impropre ------------------------------
    cibles = []
    for r in resultats:
        for h in r["halves"]:
            if not (h.get("pb_u") or h.get("pb_v") or "c" in h):
                continue
            if "c" in h:
                cc, pp = h["c"], h["p"]
            else:
                cc, pp = (h["u"], h["v"]) if h.get("pb_v") else (h["v"], h["u"])
            cibles.append((cc, pp, h.get("valeur")))
    cibles = cibles[:4]
    if cibles:
        fig, axs = plt.subplots(1, len(cibles), figsize=(5.4 * len(cibles), 4.2), squeeze=False)
        for ax, (cc, pp, val) in zip(axs[0], cibles):
            cf = float(sp.N(cc))
            if pp in (sp.oo, -sp.oo):
                sg = 1.0 if pp == sp.oo else -1.0
                d = np.concatenate([np.linspace(0, 10, 4001), np.linspace(10, 1e4, 400001)[1:]])
                X = cf + sg * d
            else:
                pf = float(sp.N(pp))
                dist = abs(cf - pf)
                d_ = np.concatenate([[0], np.geomspace(1e-9 * dist, dist, 200001)[::-1][:-1][::-1]])
                d_ = np.sort(np.unique(np.concatenate([[0], np.geomspace(1e-9 * dist, dist, 200001)])))
                X = cf + np.sign(pf - cf) * (dist - d_)  # de c vers p quand d_ décroît
                X = X[::-1]
                d = np.abs(X - cf)
            with np.errstate(all="ignore"):
                Y = fn(X)
            Y = np.where(np.isfinite(Y), Y, 0.0)
            G = np.concatenate([[0], np.cumsum(0.5 * (Y[1:] + Y[:-1]) * np.diff(X))])
            ax.semilogx(np.maximum(d, 1e-12), G, color="tab:blue", lw=1.5, label="G(X)=∫_c^X f")
            if val is not None and val not in (sp.oo, -sp.oo) and not isinstance(val, str):
                try:
                    ax.axhline(float(sp.N(val)), color="crimson", ls="--", lw=1.0, label="limite = %.5g" % float(sp.N(val)))
                except Exception:
                    pass
            ax.legend(fontsize=8)
            _style_axes(ax, "Intégrale cumulée de c=%.3g vers %s" % (cf, _nom_borne(pp).replace("\\", "")), "|X − c|", "G(X)")
        fig.tight_layout()
        st.pyplot(fig)

        # ---------- 5) comportement de |f| près du bord : droite en échelle log-log -----------------------
        fig, axs = plt.subplots(1, len(cibles), figsize=(5.4 * len(cibles), 4.2), squeeze=False)
        for ax, (cc, pp, val) in zip(axs[0], cibles):
            cf = float(sp.N(cc))
            if pp in (sp.oo, -sp.oo):
                sg = 1.0 if pp == sp.oo else -1.0
                u = np.geomspace(max(abs(cf), 1.0) * 2, 1e6, 300)
                X = sg * u
                xl_ = "|x|"
            else:
                pf = float(sp.N(pp))
                dist = abs(cf - pf)
                u = np.geomspace(1e-7 * dist, dist * 0.5, 300)
                X = pf + np.sign(cf - pf) * u
                xl_ = "|x − %.3g|" % pf
            with np.errstate(all="ignore"):
                Y = np.abs(fn(X))
            ok = np.isfinite(Y) & (Y > 0)
            if ok.sum() > 10:
                ax.loglog(u[ok], Y[ok], color="tab:blue", lw=1.5, label="|f|")
                n_ = ok.sum()
                sel = np.where(ok)[0][-int(n_ * 0.4):]
                pente = np.polyfit(np.log(u[sel]), np.log(Y[sel]), 1)
                with np.errstate(all="ignore"):
                    ajust = np.exp(pente[1]) * u[sel] ** pente[0]
                if np.all(np.isfinite(ajust)):
                    ax.loglog(u[sel], ajust, "k--", lw=1.0, label="pente %.3g (ajustement sur la fin)" % pente[0])
                ax.legend(fontsize=8)
            _style_axes(ax, "Comportement de |f| près de %s (log-log)" % _nom_borne(pp).replace("\\", ""), xl_, "|f|")
        fig.tight_layout()
        st.pyplot(fig)




# =============================================================
# Théorèmes de convergence : convergence monotone, convergence dominée, lemme de Fatou
# =============================================================
cv_limite = None    # renseigné par app.py : _cv_limite_ponctuelle
cv_fusionne = None  # renseigné par app.py : _cv_fusionne

_NS = [1, 2, 3, 4, 6, 9, 15, 30, 60, 120]


def _intervalles(ens):
    """Intervalles (non réduits à un point) qui composent un ensemble."""
    if isinstance(ens, sp.Interval):
        return [ens]
    if isinstance(ens, sp.Union):
        out = []
        for e in ens.args:
            out += _intervalles(e)
        return out
    return []


def _echs_x(lo, hi):
    if lo == -sp.oo and hi == sp.oo:
        return [-30.0, -5.0, -2.0, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.0, 5.0, 30.0]
    if lo == -sp.oo:
        h = float(sp.N(hi))
        return [h - t for t in (0.05, 0.3, 1, 3, 10, 60)]
    if hi == sp.oo:
        l = float(sp.N(lo))
        return [l + t for t in (0.05, 0.3, 1, 3, 10, 60)]
    l, h = float(sp.N(lo)), float(sp.N(hi))
    return [l + (h - l) * t for t in (0.02, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.98)]


def _val2(fmp, n_, x_):
    try:
        v = fmp(mp.mpf(n_), mp.mpf(x_))
        if isinstance(v, mp.mpc):
            if abs(v.imag) > 1e-12:
                return None
            v = v.real
        v = float(v)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _signe_global(fn, n, x, R):
    """'+' (≥0 partout), '-' (≤0 partout), '0' ou None, d'après un échantillonnage serré des (n, x)."""
    try:
        fmp = sp.lambdify((n, x), fn, modules=["mpmath"])
        mp.mp.dps = 20
        pos = neg = False
        for xv in _echs_x(R.start, R.end):
            for nv in _NS:
                v = _val2(fmp, nv, xv)
                if v is None:
                    return None
                if v > 1e-14:
                    pos = True
                elif v < -1e-14:
                    neg = True
        if pos and neg:
            return None
        return "+" if pos else ("-" if neg else "0")
    except Exception:
        return None


def _monotonie_n(fn, n, x, R):
    """'croissante', 'décroissante', 'constante' ou None."""
    try:
        D = fn.subs(n, n + 1) - fn
        fmp = sp.lambdify((n, x), D, modules=["mpmath"])
        fm_ = sp.lambdify((n, x), fn, modules=["mpmath"])
        mp.mp.dps = 20
        pos = neg = False
        for xv in _echs_x(R.start, R.end):
            for nv in _NS:
                v = _val2(fmp, nv, xv)
                w = _val2(fm_, nv, xv)
                if v is None or w is None:
                    return None
                tol = 1e-12 * (1 + abs(w))
                if v > tol:
                    pos = True
                elif v < -tol:
                    neg = True
        if pos and neg:
            return None
        return "croissante" if pos else ("décroissante" if neg else "constante")
    except Exception:
        return None


def _mesurable(fn, n, x, I):
    try:
        cd = _avec_delai(lambda: continuous_domain(fn.subs(n, 3), x, I), 15)
        if cd is not None and (cd == I or (I - cd) == S.EmptySet):
            return "continue"
        if cd is not None and isinstance(I - cd, sp.FiniteSet):
            return "continue par morceaux"
    except Exception:
        pass
    return "mesurable"


def _fn_latex_pos(fn_m):
    return sp.latex(fn_m)


# -------------------------------------------------------------
# Intégrabilité d'une fonction explicite (réutilise l'étude de ∫|g|)
# -------------------------------------------------------------
def _integrable_g(g, x, a, b, nom="g"):
    """Renvoie (verdict, lignes) pour l'intégrabilité de g sur ]a, b[."""
    xr = sp.Symbol("x", real=True)
    if x != xr:
        try:
            if a >= 0:
                xr = sp.Symbol("x", positive=True)
        except Exception:
            pass
        g2 = g.subs(x, xr)
        v, lg = _integrable_g0(g2, xr, a, b, nom)
        intro = _T("On note $f(x)=" + sp.latex(g2) + "$ la fonction $" + nom + "$ écrite en la variable $x$ et on étudie $\\int_{" + sp.latex(a) + "}^{" + sp.latex(b) + "}|f(x)|\\,dx$.")
        return v, [intro] + lg
    return _integrable_g0(g, x, a, b, nom)


def _integrable_g0(g, x, a, b, nom="g"):
    try:
        morceaux, infos = _decoupage(g, x, a, b)
    except Exception:
        return None, []
    if morceaux is None:
        return None, []
    fm = _fnum(g, x)
    lignes, verdicts = [], []
    for m in morceaux:
        if not m["pb_lo"] and not m["pb_hi"]:
            def _fini_(pt):
                try:
                    v_ = g.subs(x, pt)
                    return bool(v_.is_finite) and not v_.has(sp.nan, sp.zoo)
                except Exception:
                    return False
            if _fini_(m["lo"]) and _fini_(m["hi"]):
                lignes.append(_T("Sur $[" + sp.latex(m["lo"]) + "," + sp.latex(m["hi"]) + "]$, $"+nom+"$ est continue, donc bornée sur un segment de mesure finie : elle y est intégrable."))
            else:
                lignes.append(_T("Sur $]" + sp.latex(m["lo"]) + "," + sp.latex(m["hi"]) + "[$, $"+nom+"$ est continue et admet des limites finies aux deux bornes : elle se prolonge en une fonction continue sur le segment $[" + sp.latex(m["lo"]) + "," + sp.latex(m["hi"]) + "]$, donc bornée, donc intégrable."))
            verdicts.append(True)
    halves = _demi_integrales(morceaux)
    for (c, p) in halves:
        if len(halves) > 1:
            lignes.append(_T("Sur $" + _interv_demi(c, p) + "$ :"))
        v, lg = _abs_integrable(g, x, fm, c, p)
        lignes += lg
        verdicts.append(v)
    if not verdicts:
        return None, lignes
    if all(v is True for v in verdicts):
        return True, lignes
    if any(v is False for v in verdicts):
        return False, lignes
    return None, lignes


# -------------------------------------------------------------
# Recherche d'un majorant intégrable : sup en n
# -------------------------------------------------------------
def _sup_en_t(fn, n, x, R):
    """Majorant explicite de sup_{n>=1} |f_n(x)| par étude en la variable réelle t=n ; renvoie (g, texte) ou None."""
    t = sp.Symbol("t", positive=True)
    pair = False
    try:
        if R.start >= 0:
            xs = sp.Symbol("xs", positive=True)
        elif R.end <= 0:
            xs = sp.Symbol("xs", negative=True)
        elif sp.simplify(fn.subs(x, -x) - fn) == 0:
            xs = sp.Symbol("xs", positive=True)
            pair = True
        else:
            xs = sp.Symbol("xs", real=True)
    except Exception:
        xs = sp.Symbol("xs", real=True)
    fx = fn.subs(x, xs)
    g_ = fx.subs(n, t)
    dg = _avec_delai(lambda: sp.simplify(sp.diff(g_, t)), 15)
    if dg is None:
        return None
    sols = _avec_delai(lambda: sp.solve(sp.Eq(dg, 0), t), 15)
    if not sols:
        return None
    sols = [s for s in sols if s.is_real is not False]
    if len(sols) != 1:
        return None
    tc = sp.simplify(sols[0])
    if tc.has(sp.I):
        return None
    val = sp.simplify(fx.subs(n, tc))
    # maximum global en t > 0 : valeur critique supérieure aux limites en 0+ et +oo (testé en des x rationnels de R)
    try:
        pts_ = [e_ for e_ in _echs_x(R.start, R.end) if R.contains(e_) == True and e_ != 0]
        pts_ = [sp.nsimplify(abs(e_) if pair else e_) for e_ in pts_][:5]
        if not pts_:
            return None
        for xv in pts_:
            gv = g_.subs(xs, xv)
            vv = val.subs(xs, xv)
            for pt in (0, sp.oo):
                lim_ = _avec_delai(lambda: sp.limit(gv, t, pt, "+" if pt == 0 else None), 8)
                if lim_ is None or lim_.has(sp.Limit):
                    return None
                dif_ = sp.N(sp.Abs(vv) - sp.Abs(lim_))
                if dif_ < -1e-9:
                    return None
    except Exception:
        return None
    if pair:
        return val.subs(xs, sp.Abs(x)), tc.subs(xs, sp.Abs(x))
    return val.subs(xs, x), tc.subs(xs, x)


def _borne_constante(fn, n, x, R):
    """Majorant constant M de |f_n| (n >= 1) sur R, si le maximum en t est indépendant de x. Renvoie (M, texte) ou None."""
    r = _sup_en_t(fn, n, x, R)
    if r is None:
        return None
    val, tc = r
    if val.has(x):
        return None
    return sp.simplify(sp.Abs(val)), tc


def _facteurs_bornes(fn, n, x, I):
    """f_n = h(x) * r(n, x) avec |r| <= M : renvoie (h, M, justification) ou None."""
    h, rest = fn.as_independent(n, as_Add=False)
    if rest == 1 or not rest.has(n):
        return None
    reps = []
    M = sp.Integer(1)
    facs = list(sp.Mul.make_args(rest))
    if n in facs:
        for fc in list(facs):
            if isinstance(fc, sp.sin):
                w = sp.simplify(fc.args[0] * n)
                if not w.has(n):
                    facs.remove(n)
                    facs.remove(fc)
                    reps.append(r"|\sin t|\le |t| \text{ donne } |" + sp.latex(n * fc) + r"|\le |" + sp.latex(w) + r"|")
                    h = h * w
                    break
    for fac in facs:
        if fac == 1:
            continue
        if isinstance(fac, (sp.sin, sp.cos)):
            reps.append(r"|" + sp.latex(fac) + r"|\le 1")
            continue
        if fac.is_Pow and fac.exp.has(n) and not fac.base.has(n):
            u = fac.base
            ok = None
            for rg in [sp.Interval(I.start, I.end)]:
                vals = []
                for xv in _echs_x(I.start, I.end):
                    try:
                        vals.append(float(sp.N(u.subs(x, xv))))
                    except Exception:
                        vals.append(None)
                if all(v is not None for v in vals) and all(-1e-12 <= v <= 1 + 1e-12 for v in vals):
                    ok = True
            if ok and (u.is_polynomial(x) or u.is_rational_function(x)):
                reps.append(r"0\le " + sp.latex(u) + r"\le 1 \text{ sur } I \Rightarrow 0\le \left(" + sp.latex(u) + r"\right)^{n}\le 1")
                continue
        if isinstance(fac, sp.exp):
            arg = fac.args[0]
            u = sp.simplify(-arg / n)
            if not u.has(n):
                vals = []
                for xv in _echs_x(I.start, I.end):
                    try:
                        vals.append(float(sp.N(u.subs(x, xv))))
                    except Exception:
                        vals.append(None)
                if all(v is not None for v in vals) and all(v >= -1e-12 for v in vals):
                    reps.append(r"0\le e^{" + sp.latex(arg) + r"}\le 1 \text{ car } " + sp.latex(u) + r"\ge 0")
                    continue
        if fac.is_Pow and fac.base.has(x) and not fac.base.has(n) and fac.exp.has(n):
            kk = sp.simplify(fac.exp * n)
            if kk.is_number and not kk.has(n) and kk != 0:
                ub = fac.base
                try:
                    pos = all(float(sp.N(ub.subs(x, xv))) > 0 for xv in _echs_x(I.start, I.end))
                except Exception:
                    pos = False
                if pos:
                    reps.append(sp.latex(fac) + r"\le \max(1," + sp.latex(ub) + r")^{" + sp.latex(abs(kk)) + r"}\le (1+" + sp.latex(ub) + r")^{" + sp.latex(abs(kk)) + r"}")
                    h = h * (1 + ub) ** abs(kk)
                    continue
        if fac.is_Pow and fac.base == n and fac.exp.is_number and fac.exp < 0:
            reps.append(r"0<" + sp.latex(fac) + r"\le 1 \text{ pour } n\ge1")
            continue
        # facteur plus complexe : maximum en la variable n, indépendant de x
        bc = _borne_constante(fac, n, x, I)
        if bc is not None:
            Mi, tc = bc
            reps.append(r"\sup_{t>0}\left|" + sp.latex(fac.subs(n, sp.Symbol('t'))) + r"\right|=" + sp.latex(Mi) + r"\ (\text{atteint pour } t=" + sp.latex(tc) + r")")
            M *= Mi
            continue
        return None
    return h, M, reps


def _majorant(fn, n, x, I, a, b, f_lim, monot):
    """Cherche un majorant g intégrable de |f_n| (tous n). Renvoie (g, lignes_justif, verdict_integrabilite, lignes_integrabilite)."""
    essais = []
    try:
        num_, den_ = sp.fraction(sp.together(fn))
        dif = sp.simplify(den_ - num_)
        if dif.has(n) and dif.is_number is False and False:
            pass
        ech_ = _echs_x(I.start, I.end)
        if num_ != 1 and den_.has(n) and dif.is_number and dif > 0 and all(float(sp.N(num_.subs({x: xv, n: kk}))) >= 0 and float(sp.N(den_.subs({x: xv, n: kk}))) > 0 for xv in ech_ for kk in (1, 2, 5)):
            g1 = sp.Integer(1)
            just1 = [_T("Pour tout $n$ et tout $x\\in I$ : $0\\le " + sp.latex(num_) + "\\le " + sp.latex(den_) + "$ (car $" + sp.latex(den_) + "-" + sp.latex(sp.simplify(num_)) + "=" + sp.latex(dif) + ">0$), donc $0\\le f_n(x)\\le g(x)=1$.")]
            essais.append((g1, just1))
    except Exception:
        pass
    fb = _facteurs_bornes(fn, n, x, I)
    if fb is not None:
        h, M, reps = fb
        g = sp.simplify(M * sp.Abs(h)) if not h.is_positive else sp.simplify(M * h)
        try:
            if all(float(sp.N(h.subs(x, xv))) >= -1e-12 for xv in _echs_x(I.start, I.end)):
                g = sp.simplify(M * h)
        except Exception:
            pass
        hl = "-" if h == -1 else ("" if h == 1 else sp.latex(h) + "\\cdot ")
        just = [_T("Écrivons $f_n(x)=" + hl + "r_n(x)$ avec $r_n(x)=" + sp.latex(sp.simplify(fn / h)) + "$. Pour tout $n$ : $" + ",\\ ".join(reps) + "$, d'où $|f_n(x)|\\le g(x)=" + sp.latex(g) + "$.")]
        essais.append((g, just))
    if monot in ("croissante", "décroissante"):
        pass
    for (g, just) in essais:
        v, lg = _integrable_g(g, x, a, b)
        if v is True:
            return g, just, v, lg
    if essais:
        g, just = essais[0]
        v, lg = _integrable_g(g, x, a, b)
        return g, just, v, lg
    r = _sup_en_t(fn, n, x, I)
    if r is not None:
        val, tc = r
        g = sp.simplify(sp.Abs(val))
        just = [_T("Pour $x$ fixé, la fonction $t\\mapsto f_t(x)$ atteint son maximum en $t=" + sp.latex(tc) + "$, d'où $|f_n(x)|\\le g(x)=" + sp.latex(g) + "$ pour tout $n$.")]
        v, lg = _integrable_g(g, x, a, b)
        return g, just, v, lg
    return None, None, None, None


# -------------------------------------------------------------
# Intégrales ∫ f_n et ∫ f
# -------------------------------------------------------------
def _quad_n(fn, n, x, nv, a, b):
    try:
        fm = _fnum(fn.subs(n, nv), x)
        return _quad(fm, a, b)
    except Exception:
        return None


def _lx(e):
    if e == sp.oo:
        return "+\\infty"
    return sp.latex(e)


def _court(e):
    cands = [e]
    for f_ in (sp.simplify, sp.expand, sp.factor, sp.powsimp, lambda t: sp.simplify(sp.expand(t))):
        try:
            cands.append(f_(e))
        except Exception:
            pass
    return min(cands, key=lambda t: sp.count_ops(t))


def _int_n(fn_m, n1, x, a, b):
    """Calcule I_n = ∫ f_n (formule en n) : renvoie dict(F, In, L, lignes)."""
    res = dict(F=None, In=None, L=None, lignes=[])
    F = _avec_delai(lambda: sp.integrate(fn_m, x), 12)
    def _branche(P_):
        if not isinstance(P_, sp.Piecewise):
            return P_, 1
        for e_, c_ in P_.args:
            try:
                if all(bool(c_.subs(n1, k)) for k in (4, 9, 15)):
                    n0_ = 4
                    for k0 in (3, 2, 1):
                        if bool(c_.subs(n1, k0)):
                            n0_ = k0
                        else:
                            break
                    return e_, n0_
            except Exception:
                pass
        return P_, 1
    if F is not None and isinstance(F, sp.Piecewise):
        F, _n0F = _branche(F)
        res["n0"] = _n0F
    _SPEC = (sp.Integral, sp.meijerg, sp.hyper, sp.Piecewise, sp.lerchphi, sp.polylog, sp.uppergamma, sp.lowergamma, sp.Ei, sp.li)
    if F is not None and F.has(*_SPEC):
        F = None
    if F is not None:
        try:
            Fa = _valeur_F(F, x, a, "+")
            Fb = _valeur_F(F, x, b, "-")
            if Fa[0] == "fin" and Fb[0] == "fin":
                res["F"] = F
                res["In"] = _court(Fb[1] - Fa[1])
                res["bornes"] = (Fa[1], Fb[1])
        except Exception:
            pass
    if res["In"] is None:
        In = _avec_delai(lambda: sp.integrate(fn_m, (x, a, b)), 15)
        if In is not None and isinstance(In, sp.Piecewise):
            In, res["n0"] = _branche(In)
            In = sp.gammasimp(In) if In.has(sp.gamma) else In
        if In is not None and not In.has(sp.Integral, sp.meijerg, sp.hyper, sp.lerchphi, sp.polylog, sp.uppergamma, sp.lowergamma, sp.Ei, sp.li):
            if False:
                for e_, c_ in In.args:
                    try:
                        if all(bool(c_.subs(n1, k)) for k in (4, 9, 15)):
                            In = e_
                            for k0 in (1, 2, 3):
                                if bool(c_.subs(n1, k0)):
                                    res["n0"] = k0
                                    break
                            else:
                                res["n0"] = 4
                                for k0 in (3, 2, 1):
                                    if not bool(c_.subs(n1, k0)):
                                        res["n0"] = k0 + 1
                                        break
                            break
                    except Exception:
                        pass
            if not isinstance(In, sp.Piecewise):
                res["In"] = _court(In)
    if res["In"] is not None:
        res["L"] = _avec_delai(lambda: sp.limit(res["In"], n1, sp.oo), 20)
        if res["L"] is not None and res["L"].has(sp.Limit):
            res["L"] = None
        elif res["L"] is not None:
            res["L"] = _court(res["L"])
    return res


def _valeur_f(groupes, x):
    """Intégrale de la limite simple et expression de f (Piecewise) à partir des groupes convergents non négligeables."""
    morceaux, total = [], 0
    for g in groupes:
        for iv in _intervalles(g["ens"]):
            morceaux.append((iv, g["lim"]))
    exprs = []
    for iv, lim in morceaux:
        try:
            cond = iv.as_relational(x)
        except Exception:
            cond = True
        exprs.append((lim, cond))
    f_lim = sp.Piecewise(*exprs) if len(exprs) > 1 else (exprs[0][0] if exprs else sp.Integer(0))
    if len(exprs) > 1:
        f_lim = sp.Piecewise(*exprs, (0, True))
    J = 0
    for iv, lim in morceaux:
        v = _avec_delai(lambda: sp.integrate(lim, (x, iv.start, iv.end)), 20)
        if v is None or v.has(sp.Integral):
            return f_lim, None
        J += v
    return f_lim, sp.simplify(J)


def _limite_dernier_mot(L):
    if L is None:
        return None
    return L


def _lat_ens(I):
    if isinstance(I, sp.Interval):
        return _iv(I.start, I.end, I.left_open, I.right_open)
    if isinstance(I, sp.FiniteSet):
        return "\\{" + ",".join(sp.latex(a_) for a_ in sorted(I.args, key=lambda t_: float(t_))) + "\\}"
    if isinstance(I, sp.Union):
        parts = sorted(I.args, key=lambda t_: float(t_.start if isinstance(t_, sp.Interval) else t_.args[0]))
        return " \\cup ".join(_lat_ens(p_) for p_ in parts)
    return sp.latex(I)


def _lat_f(groupes, x):
    """Écriture de la limite simple : une seule expression, ou 'cases' selon les ensembles."""
    gs = [g for g in groupes if g["statut"] == "cv"]
    if len(gs) == 1:
        return sp.latex(gs[0]["lim"])
    lignes = []
    for g in gs:
        e_ = g["ens"]
        cond = (r"x=" + sp.latex(e_.args[0])) if (isinstance(e_, sp.FiniteSet) and len(e_.args) == 1) else (r"x\in " + _lat_ens(e_))
        lignes.append(sp.latex(g["lim"]) + r" & \text{si } " + cond)
    return r"\begin{cases}" + r"\\ ".join(lignes) + r"\end{cases}"



# -------------------------------------------------------------
# Limite simple : repli robuste (raisons géométriques, Min/Max, signe de x)
# -------------------------------------------------------------
def _bases_geo(fn, n):
    res = []
    for p_ in fn.atoms(sp.Pow):
        if p_.exp.has(n) and not p_.base.has(n):
            k = sp.simplify(p_.exp / n)
            if k.is_number and k > 0:
                res.append((p_, p_.base, k))
    return res


def _lim_valeur(fn, n, x, xv):
    """Limite quand n -> oo de f_n(xv) (xv réel fixé) ; None si échec."""
    m = sp.Symbol("m", positive=True)
    try:
        e = fn.subs(x, xv).subs(n, m)
        e = e.replace(lambda t: isinstance(t, (sp.Min, sp.Max)), lambda t: type(t)(*[sp.limit(a, m, sp.oo) for a in t.args]))
        L0 = sp.limit(e, m, sp.oo)
        if L0.has(sp.Limit) or L0.has(m) or L0 in (sp.zoo, sp.nan):
            return None
        return sp.nsimplify(sp.simplify(L0)) if L0.is_number else sp.simplify(L0)
    except Exception:
        return None


def _lim_regime(fn, n, x, iv):
    """Limite sur l'intervalle ouvert iv, avec régimes |u|<1 / |u|>1 pour les puissances u^n."""
    m = sp.Symbol("m", positive=True)
    ech = _echs_x(iv.start, iv.end)
    ech = [e_ for e_ in ech if iv.contains(e_) == True] or [ (iv.start + iv.end) / 2 if iv.start.is_finite and iv.end.is_finite else (iv.start + 1 if iv.start.is_finite else iv.end - 1)]
    xm = ech[len(ech) // 2]
    Q = sp.Symbol("Q", positive=True)
    e = fn
    textes = []
    for (p_, b, k) in _bases_geo(fn, n):
        try:
            vb = abs(float(sp.N(b.subs(x, xm))))
        except Exception:
            return None
        if abs(vb - 1) < 1e-9:
            return None
        e = e.subs(p_, Q ** k if k != 1 else Q)
        textes.append((b, vb < 1))
    if not textes:
        return None
    if len({t[1] for t in textes}) != 1:
        return None
    petit = textes[0][1]
    try:
        L0 = sp.limit(e, Q, 0 if petit else sp.oo, "+" if petit else "-")
        if L0.has(sp.Limit) or L0.has(Q) or L0 in (sp.zoo, sp.nan, sp.oo, -sp.oo):
            return None
        return sp.simplify(L0), textes
    except Exception:
        return None


def _verif_lim(fn, n, x, L, iv):
    try:
        pts = [e_ for e_ in _echs_x(iv.start, iv.end) if iv.contains(e_) == True][:4] if isinstance(iv, sp.Interval) else [iv]
        if not pts:
            return False
        for xv in pts:
            ref = complex(sp.N(L.subs(x, xv))) if hasattr(L, "subs") else complex(L)
            err = []
            for N_ in (2000, 200000, 20000000):
                v = complex(sp.N(fn.subs({x: xv, n: N_}), 30))
                err.append(abs(v - ref))
            tol = 5e-3 * (1 + abs(ref))
            if not (err[2] <= tol or (err[2] <= 0.5 * err[1] and err[1] <= 0.5 * err[0] + 1e-12)):
                return False
        return True
    except Exception:
        return False


def _reparer_limites(fn_app, n_app, x, pieces):
    """Complète les pièces 'ind' (ou suspectes) ; renvoie la nouvelle liste de pièces."""
    neufs = []
    for pc in pieces:
        ens = pc["ens"]
        suspect = pc["statut"] == "ind" or (pc["lim"] is not None and pc["lim"].has(sp.oo, -sp.oo, sp.zoo, sp.nan))
        if pc["statut"] == "cv" and pc["lim"] is not None and not suspect:
            try:
                pc["lim"] = sp.simplify(pc["lim"])
                pc["ligne"] = pc["ligne"].replace("arctan(0)", "0")
            except Exception:
                pass
        if not suspect:
            neufs.append(pc)
            continue
        if isinstance(ens, sp.FiniteSet):
            p_ = ens.args[0]
            L0 = _lim_valeur(fn_app, n_app, x, p_)
            if L0 is not None and _verif_lim(fn_app, n_app, x, sp.sympify(L0), p_):
                fx = sp.simplify(fn_app.subs(x, p_))
                neufs.append(dict(ens=ens, statut="cv", lim=L0, ligne=r"\text{Pour } x=" + sp.latex(p_) + r",\quad f_n(" + sp.latex(p_) + r")=" + sp.latex(fx) + r"\ \longrightarrow\ " + sp.latex(L0)))
            else:
                neufs.append(pc)
            continue
        if not isinstance(ens, sp.Interval):
            neufs.append(pc)
            continue
        # points de rupture de régime : |u| = 1
        pts = []
        for (p_, b, k) in _bases_geo(fn_app, n_app):
            try:
                sol = sp.solveset(sp.Eq(sp.Abs(b), 1), x, domain=ens)
                if isinstance(sol, sp.FiniteSet):
                    pts += [q for q in sol.args if ens.contains(q) == True and q != ens.start and q != ens.end]
            except Exception:
                pass
        pts = sorted(set(pts), key=lambda t_: float(t_))
        bornes = [ens.start] + pts + [ens.end]
        sous = []
        ok = True
        for k in range(len(bornes) - 1):
            iv = sp.Interval.open(bornes[k], bornes[k + 1])
            res = _lim_regime(fn_app, n_app, x, iv)
            if res is not None:
                L0, textes = res
                if not _verif_lim(fn_app, n_app, x, L0, iv):
                    ok = False
                    break
                b0 = textes[0][0]
                if textes[0][1]:
                    expl = r"\left|" + sp.latex(b0) + r"\right|<1 \text{, donc } (" + sp.latex(b0) + r")^{n}\to0"
                else:
                    expl = r"\left|" + sp.latex(b0) + r"\right|>1 \text{, donc } \left|(" + sp.latex(b0) + r")^{n}\right|\to+\infty"
                sous.append(dict(ens=iv, statut="cv", lim=L0, ligne=r"\text{Pour } x\in " + sp.latex(iv) + r",\quad " + expl + r"\text{ et } f_n(x)\longrightarrow " + sp.latex(L0)))
                continue
            # signe de x fixé, limite directe
            try:
                if iv.start >= 0:
                    d = sp.Symbol("d", positive=True)
                elif iv.end <= 0:
                    d = sp.Symbol("d", negative=True)
                else:
                    ok = False
                    break
                m = sp.Symbol("m", positive=True)
                e = fn_app.subs(x, d).subs(n_app, m)
                e = e.replace(lambda t: isinstance(t, (sp.Min, sp.Max)), lambda t: type(t)(*[sp.limit(a, m, sp.oo) for a in t.args]))
                L0 = sp.limit(e, m, sp.oo)
                if L0.has(sp.Limit, m) or L0 in (sp.oo, -sp.oo, sp.zoo, sp.nan):
                    ok = False
                    break
                L0 = sp.simplify(L0.subs(d, x))
                if not _verif_lim(fn_app, n_app, x, L0, iv):
                    ok = False
                    break
                sous.append(dict(ens=iv, statut="cv", lim=L0, ligne=r"\text{Pour } x\in " + sp.latex(iv) + r",\quad f_n(x)=" + sp.latex(fn_app) + r"\ \longrightarrow\ " + sp.latex(L0)))
            except Exception:
                ok = False
                break
        if not ok:
            neufs.append(pc)
            continue
        for k, sub in enumerate(sous):
            neufs.append(sub)
            if k < len(pts):
                p_ = pts[k]
                L0 = _lim_valeur(fn_app, n_app, x, p_)
                if L0 is not None:
                    fx = sp.simplify(fn_app.subs(x, p_))
                    neufs.append(dict(ens=sp.FiniteSet(p_), statut="cv", lim=L0, ligne=r"\text{Pour } x=" + sp.latex(p_) + r",\quad f_n(" + sp.latex(p_) + r")=" + sp.latex(fx) + r"\ \longrightarrow\ " + sp.latex(L0)))
                else:
                    neufs.append(dict(ens=sp.FiniteSet(p_), statut="ind", lim=None, ligne=r"\text{Pour } x=" + sp.latex(p_) + r",\quad \text{la limite n'a pas pu être déterminée.}"))
    return neufs


# -------------------------------------------------------------
# Étude complète
# -------------------------------------------------------------
def etude_theoremes_convergence(f_expr, n_app, x, intervalle, N=10):
    _VUES.clear()
    _GAUSS_Q.clear()
    _ROT.clear()
    _REFS.clear()
    P = _P
    n1 = sp.Symbol("n", integer=True, positive=True)
    try:
        fn_app = sp.sympify(f_expr)
    except Exception:
        _rouge("Expression invalide pour f_n.")
        return
    if intervalle == S.Reals or not isinstance(intervalle, sp.Interval):
        I = sp.Interval(-sp.oo, sp.oo, True, True)
    else:
        I = intervalle
    a, b = I.start, I.end
    if I.is_EmptySet or a == b:
        _rouge("L'intervalle est vide ou réduit à un point.")
        return
    fn = fn_app.subs(n_app, n1)
    if not fn.has(n1):
        _rouge("La suite doit dépendre de n (la variable est n).")
        return
    latI = _lat_ens(I)

    _titre("Données")
    P(_T("Suite de fonctions $f_n(x)=" + sp.latex(fn) + "$ sur $I=" + latI + "$, avec $n\\ge1$. On étudie le passage à la limite sous l'intégrale : $\\lim_{n\\to+\\infty}\\int_I f_n\\,d\\lambda$ et $\\int_I\\lim_{n\\to+\\infty} f_n\\,d\\lambda$."))
    nature = _mesurable(fn, n1, x, I)
    P(_T("Pour tout $n$, $f_n$ est " + ("continue" if nature == "continue" else ("continue par morceaux" if nature == "continue par morceaux" else "mesurable")) + " sur $I$, donc mesurable."))

    # ---- limite simple -----------------------------------------------------------------
    _titre("Limite simple")
    if cv_limite is None:
        pieces = [dict(ens=I, statut="cv", lim=sp.limit(fn_app, n_app, sp.oo), ligne=r"\text{Pour } x\in " + latI + r"\text{, } f_n(x)\to " + sp.latex(sp.limit(fn_app, n_app, sp.oo)))]
        groupes = [dict(ens=I, statut="cv", lim=pieces[0]["lim"])]
    else:
        pieces = _avec_delai(lambda: cv_limite(fn_app, n_app, x, I), 60) or []
        groupes = cv_fusionne(pieces) if (cv_fusionne is not None and pieces) else []
    if cv_limite is not None:
        pieces = _reparer_limites(fn_app, n_app, x, pieces)
        for pc in pieces:
            if pc["statut"] == "ind" and isinstance(pc["ens"], sp.Interval):
                iv = pc["ens"]
                try:
                    for xv in [e_ for e_ in _echs_x(iv.start, iv.end) if iv.contains(e_) == True][:3]:
                        vals = [float(sp.N(fn_app.subs({x: xv, n_app: k}), 20)) for k in range(3000, 3011)]
                        if max(vals) - min(vals) > 0.05 * (1 + max(abs(v_) for v_ in vals)) * 0.2:
                            pc["statut"] = "div"
                            pc["lim"] = None
                            pc["ligne"] = r"\text{Pour } x\in " + sp.latex(iv) + r",\quad \text{la suite } (f_n(x))_n \text{ oscille sans se stabiliser : elle n'admet pas de limite.}"
                            break
                except Exception:
                    pass
        if fn_app.is_Pow and fn_app.exp == n_app and not fn_app.base.has(n_app):
            u = fn_app.base
            neufs = []
            for pc in pieces:
                suspect = pc["statut"] != "cv" or (pc["lim"] is not None and pc["lim"].has(sp.oo, -sp.oo, sp.zoo, sp.nan))
                if not (suspect and isinstance(pc["ens"], sp.Interval)):
                    neufs.append(pc)
                    continue
                iv = pc["ens"]
                try:
                    sol = sp.solveset(sp.Eq(sp.Abs(u), 1), x, domain=iv)
                    pts = sorted([p_ for p_ in sol.args if iv.contains(p_) == True and p_ != iv.start and p_ != iv.end], key=lambda t_: float(t_)) if isinstance(sol, sp.FiniteSet) else None
                except Exception:
                    pts = None
                if pts is None:
                    neufs.append(pc)
                    continue
                bornes = [iv.start] + pts + [iv.end]
                for k in range(len(bornes) - 1):
                    sub = sp.Interval.open(bornes[k], bornes[k + 1])
                    neufs.append(dict(ens=sub, statut="cv", lim=sp.Integer(0), ligne=r"\text{Pour } x\in " + sp.latex(sub) + r",\quad \left|" + sp.latex(u) + r"\right|<1 \text{ : } f_n(x)=\left(" + sp.latex(u) + r"\right)^{n} \text{ est une suite géométrique de raison de valeur absolue } <1\text{, donc } f_n(x)\longrightarrow 0"))
                for p_ in pts:
                    v_ = sp.simplify(u.subs(x, p_))
                    if v_ == 1:
                        neufs.append(dict(ens=sp.FiniteSet(p_), statut="cv", lim=sp.Integer(1), ligne=r"\text{Pour } x=" + sp.latex(p_) + r",\quad f_n(x)=1^{n}=1\ \longrightarrow\ 1"))
                    else:
                        neufs.append(dict(ens=sp.FiniteSet(p_), statut="div", lim=None, ligne=r"\text{Pour } x=" + sp.latex(p_) + r",\quad f_n(x)=(-1)^{n} \text{ n'a pas de limite.}"))
            pieces = neufs
        groupes = cv_fusionne(pieces)
    if not groupes:
        _rouge("La limite simple n'a pas pu être déterminée.")
        return
    for pc in pieces:
        P(pc["ligne"])
    nonnuls = [g for g in groupes if _intervalles(g["ens"])]
    mauvais = [g for g in nonnuls if g["statut"] != "cv"]
    if mauvais:
        P(_T("La suite $(f_n)$ ne converge pas (ou le calcul ne conclut pas) sur un ensemble de mesure non nulle : elle ne converge pas presque partout et les théorèmes de convergence ne s'appliquent pas."))
        return
    gcv = [g for g in nonnuls if g["statut"] == "cv"]
    f_lim, J = _valeur_f(gcv, x)
    gcv_all = [g for g in groupes if g["statut"] == "cv"]
    exceptionnels = [g for g in groupes if not _intervalles(g["ens"])]
    if exceptionnels:
        P(_T("Les points exceptionnels forment un ensemble fini, donc de mesure nulle : $f_n\\to f$ presque partout sur $I$, avec $f(x)=" + _lat_f(gcv_all, x) + "$."))
    else:
        P(_T("Ainsi $f_n(x)\\to f(x)=" + _lat_f(gcv_all, x) + "$ pour tout $x\\in I$, donc en particulier presque partout."))

    # ---- signe et monotonie ------------------------------------------------------------------
    regions = []
    for g in gcv:
        for iv in _intervalles(g["ens"]):
            regions.append(iv)
    signes = [_signe_global(fn, n1, x, R) for R in regions]
    monots = [_monotonie_n(fn, n1, x, R) for R in regions]
    positif = bool(signes) and all(s in ("+", "0") for s in signes)
    negatif = bool(signes) and all(s in ("-", "0") for s in signes)
    sens = None
    if monots and all(m in ("croissante", "constante") for m in monots):
        sens = "croissante"
    elif monots and all(m in ("décroissante", "constante") for m in monots):
        sens = "décroissante"
    ok_mct = False
    n0_dec = 1
    sens_p = None
    if sens is not None and (positif or negatif):
        sens_p = sens if positif else ("décroissante" if sens == "croissante" else "croissante")

    _titre("Convergence monotone")
    if not (positif or negatif):
        P(_T("La suite $(f_n)$ change de signe selon $n$ ou $x$ : les hypothèses de signe du théorème de convergence monotone ne sont pas réunies."))
    else:
        s_txt = "positives" if positif else "négatives"
        g_pos = fn if positif else -fn
        try:
            xs_ = sp.Symbol("xs", nonnegative=True)
            raison = ""
            if all(R.start >= 0 for R in regions):
                if g_pos.subs(x, xs_).is_nonnegative:
                    raison = (" (produit de facteurs positifs pour $x\\ge0$)" if positif else " (l'opposé de $f_n$ est un produit de facteurs positifs pour $x\\ge0$)")
        except Exception:
            raison = ""
        P(_T("Pour tout $n$ et tout $x\\in I$, " + ("$f_n(x)\\ge0$" if positif else "$f_n(x)\\le0$") + raison + "."))
        if sens is None:
            P(_T("En revanche la suite $(f_n(x))_n$ n'est pas monotone pour tous les $x\\in I$ : le théorème de convergence monotone ne s'applique pas."))
        else:
            # justification de la monotonie
            amgm = None
            try:
                if fn.is_Pow and fn.exp == n1:
                    gg = sp.simplify((fn.base - 1) * n1)
                    if not gg.has(n1) and all(v is not None and v >= 0 for v in [(lambda xv: float(sp.N(gg.subs(x, xv))))(xv) for xv in _echs_x(a, b)]):
                        amgm = gg
            except Exception:
                amgm = None
            try:
                rat = sp.simplify(fn.subs(n1, n1 + 1) / fn)
            except Exception:
                rat = None
            if amgm is not None:
                P(_T("Posons $u=" + sp.latex(amgm) + "\\ge0$ : l'inégalité arithmético-géométrique appliquée aux $n+1$ nombres $1+\\frac{u}{n}$ ($n$ fois) et $1$ donne $\\left(1+\\frac{u}{n}\\right)^{n}\\le\\left(1+\\frac{u}{n+1}\\right)^{n+1}$ : la suite $(f_n(x))_n$ est croissante."))
            elif rat is not None and not rat.has(n1) and not rat.has(sp.nan, sp.zoo):
                P(_T("De plus $\\frac{f_{n+1}(x)}{f_n(x)}=" + sp.latex(rat) + "$ ne dépend pas de $n$ et vaut " + ("au moins" if sens == "croissante" else "au plus") + " $1$ sur $I$ : la suite $(f_n(x))_n$ est " + sens + "."))
            else:
                try:
                    Dn = sp.factor(sp.simplify(fn.subs(n1, n1 + 1) - fn))
                    txtD = sp.latex(Dn)
                except Exception:
                    txtD = None
                if txtD and len(txtD) < 140:
                    P(_T("De plus $f_{n+1}(x)-f_n(x)=" + txtD + "$, de signe " + ("positif" if sens == "croissante" else "négatif") + " sur $I$ : la suite $(f_n(x))_n$ est " + sens + "."))
                else:
                    P(_T("L'étude du signe de $f_{n+1}-f_n$ montre que la suite $(f_n(x))_n$ est " + sens + " pour tout $x\\in I$."))
            lab = "f_n" if positif else "-f_n"
            lab1 = "f_1" if positif else "-f_1"
            if sens_p == "croissante":
                ok_mct = True
                P(_T("Les $" + lab + "$ sont mesurables, positives et croissantes en $n$ : le théorème de convergence monotone donne $\\int_I " + lab + "\\,d\\lambda\\nearrow\\int_I " + ("f" if positif else "(-f)") + "\\,d\\lambda$, limite éventuellement infinie."))
            else:
                v1, lg1, k1 = None, [], 1
                for k1 in (1, 2, 3, 4):
                    v1, lg1 = _integrable_g(g_pos.subs(n1, k1), x, a, b, ("" if positif else "-") + "f_{%d}" % k1)
                    if v1 is True:
                        break
                n0_dec = k1
                lab1 = ("" if positif else "-") + "f_{%d}" % k1
                if v1 is True:
                    ok_mct = True
                    for l_ in lg1:
                        P(l_)
                    rang = "" if k1 == 1 else " à partir du rang $n\\ge%d$" % k1
                    P(_T("Les $" + lab + "$ sont mesurables, positives, décroissantes en $n$" + rang + " et $" + lab1 + "$ est intégrable : le théorème de convergence monotone (version décroissante) donne $\\int_I " + lab + "\\,d\\lambda\\searrow\\int_I " + ("f" if positif else "(-f)") + "\\,d\\lambda$" + (" pour $n\\ge%d$" % k1 if k1 > 1 else "") + "."))
                else:
                    P(_T("Les $" + lab + "$ décroissent, mais aucun $" + "f_{k}" + "$ ne se révèle intégrable sur $I$ pour les premiers rangs : la version décroissante du théorème de convergence monotone ne s'applique pas."))

    # ---- convergence dominée ---------------------------------------------------------------------
    _titre("Convergence dominée")
    ok_dom = False
    g_maj = None
    J_fini = J is not None and not J.has(sp.oo, -sp.oo, sp.zoo, sp.nan)
    if ok_mct and sens_p == "croissante" and J_fini:
        g_maj = sp.Abs(f_lim) if positif else sp.Abs(f_lim)
        ok_dom = True
        P(_T("Comme $|f_n|\\le|f|$ pour tout $n$ et que $f$ est intégrable sur $I$ ($\\int_I f\\,d\\lambda=" + _lx(J) + "$), la suite est dominée par $g=|f|$ : le théorème de convergence dominée s'applique aussi."))
    elif ok_mct and sens_p == "décroissante":
        ok_dom = True
        g_maj = sp.Abs(fn.subs(n1, n0_dec))
        P(_T("Comme $|f_n|\\le|f_{%d}|$ pour $n\\ge%d$ avec $f_{%d}$ intégrable, la suite est dominée par $g=|f_{%d}|$ : le théorème de convergence dominée s'applique aussi." % (n0_dec, n0_dec, n0_dec, n0_dec)))
    elif ok_mct and sens_p == "croissante" and J is not None:
        P(_T("La limite $f$ n'est pas intégrable sur $I$ : une domination $|f_n|\\le g$ avec $g\\in L^1(I)$ entraînerait $|f|\\le g$ presque partout, donc $f\\in L^1(I)$. Le théorème de convergence dominée est donc exclu."))
    else:
        gm, just, vint, lgint = _majorant(fn, n1, x, I, a, b, f_lim, sens)
        if gm is None:
            P(_T("Aucun majorant explicite de $\\sup_n|f_n|$ n'a été trouvé : on ne peut pas invoquer le théorème de convergence dominée à partir de la seule expression de $f_n$."))
        else:
            for l_ in just:
                P(l_)
            if vint is True:
                for l_ in lgint:
                    P(l_)
                ok_dom = True
                g_maj = gm
                P(_T("Les $f_n$ sont mesurables, $f_n\\to f$ presque partout et $|f_n|\\le g$ avec $g$ intégrable sur $I$ : le théorème de convergence dominée s'applique."))
            else:
                for l_ in lgint or []:
                    P(l_)
                P(_T("Ce majorant n'est pas intégrable sur $I$ : l'hypothèse de domination n'est pas établie."))
    if ok_dom:
        P(_T("Il donne $f\\in L^1(I)$, $\\lim_{n\\to+\\infty}\\int_I f_n\\,d\\lambda=\\int_I f\\,d\\lambda$ et même $\\int_I|f_n-f|\\,d\\lambda\\to0$."))

    # ---- calcul direct --------------------------------------------------------------------------------
    _titre("Calcul direct de ∫ f_n")
    R_ = _int_n(fn, n1, x, a, b)
    In, L = R_["In"], R_["L"]
    if R_["F"] is not None:
        pbu = a in (sp.oo, -sp.oo)
        pbv = b in (sp.oo, -sp.oo)
        bloc_n = _chaine_int(fn, x, a, b, pbu, pbv, In) if not (pbu and pbv) else None
        if bloc_n:
            P(bloc_n)
        else:
            lg_n = _narrer_primitive(fn, x, R_["F"])
            for l_ in lg_n:
                P(l_)
        Fa, Fb = R_["bornes"]
        P(_T("Donc $\\int_I f_n\\,d\\lambda=" + sp.latex(In) + "$ pour tout $n\\ge%d$." % R_.get("n0", 1)))
    elif In is not None:
        P(_T("Un calcul direct donne $\\int_I f_n\\,d\\lambda=" + sp.latex(In) + "$ pour tout $n\\ge%d$." % R_.get("n0", 1)))
    else:
        P(_T("L'intégrale $\\int_I f_n\\,d\\lambda$ ne se met pas sous forme fermée en $n$."))
        try:
            vq = []
            for kk in (1, 10, 100):
                q_ = _quad_n(fn, n1, x, kk, a, b)
                if q_ is not None and q_ == q_:
                    vq.append("n=%d : %.4f" % (kk, float(q_)))
            if len(vq) >= 2:
                P(_T("À titre de simple contrôle numérique (sans valeur de preuve), $\\int_I f_n\\,d\\lambda$ vaut environ " + " ; ".join(vq) + "."))
        except Exception:
            pass
    if J is not None:
        try:
            if f_lim is not None and not f_lim.has(sp.Piecewise, sp.Abs) and not f_lim.is_number and J not in (sp.oo, -sp.oo) and not (a in (sp.oo, -sp.oo) and b in (sp.oo, -sp.oo)):
                bloc_j = _chaine_int(f_lim, x, a, b, a in (sp.oo, -sp.oo), b in (sp.oo, -sp.oo), J)
                if bloc_j:
                    P(bloc_j)
        except Exception:
            pass
        P(_T("Par ailleurs $\\int_I f\\,d\\lambda=" + _lx(J) + "$."))
    egal = None
    if L is not None and J is not None:
        P(_T("Quand $n\\to+\\infty$, $\\int_I f_n\\,d\\lambda\\to " + _lx(L) + "$."))
        try:
            if L.has(sp.oo, -sp.oo) or J.has(sp.oo, -sp.oo):
                egal = bool(L == J)
            else:
                egal = bool(sp.simplify(L - J) == 0)
        except Exception:
            egal = None
        if egal and (L.has(sp.oo, -sp.oo)):
            P(_T("Les deux membres valent $" + _lx(L) + "$ : l'égalité $\\lim\\int_I f_n=\\int_I\\lim f_n$ a bien lieu dans la droite achevée, comme l'annonce le théorème de convergence monotone."))
        elif egal:
            if ok_mct or ok_dom:
                P(_T("On retrouve $\\lim\\int_I f_n=\\int_I\\lim f_n=" + _lx(J) + "$, comme l'annonçait le théorème."))
            else:
                P(_T("On a bien $\\lim\\int_I f_n=\\int_I\\lim f_n$ ici, mais cette égalité ne résulte d'aucun des deux théorèmes : il s'agit d'une vérification directe."))
        else:
            P(_T("Ainsi $\\lim\\int_I f_n\\,d\\lambda=" + _lx(L) + "\\ne " + _lx(J) + "=\\int_I\\lim f_n\\,d\\lambda$ : on ne peut pas permuter limite et intégrale."))
            P(_T("Par contraposée du théorème de convergence dominée, il n'existe aucune fonction intégrable majorant tous les $|f_n|$."))
            if positif:
                try:
                    cmp_ = "\\le" if sp.simplify(J - L).is_nonpositive else "\\ge"
                except Exception:
                    cmp_ = "\\le"
                P(_T("Le lemme de Fatou ($f_n\\ge0$ mesurables) assure seulement $\\int_I\\liminf f_n\\,d\\lambda\\le\\liminf\\int_I f_n\\,d\\lambda$, soit ici $" + _lx(J) + cmp_ + _lx(L) + "$ ; l'inégalité est stricte : une masse égale à $" + sp.latex(sp.simplify(L - J)) + "$ ne se retrouve pas dans la limite simple."))
    elif L is not None:
        P(_T("Quand $n\\to+\\infty$, $\\int_I f_n\\,d\\lambda\\to " + _lx(L) + "$."))

    # ---- bilan ---------------------------------------------------------------------------------------------
    _titre("Bilan")
    lignes_b = []
    lignes_b.append("convergence monotone : " + ("applicable" if ok_mct else "non applicable"))
    lignes_b.append("convergence dominée : " + ("applicable" if ok_dom else "non applicable"))
    P(_T("Théorème de " + lignes_b[0] + " ; théorème de " + lignes_b[1] + "."))
    if (ok_mct or ok_dom) and J is not None:
        P(_T("Conclusion : $\\lim_{n\\to+\\infty}\\int_I f_n\\,d\\lambda=\\int_I f\\,d\\lambda=" + _lx(J) + "$."))
    elif egal and J is not None:
        P(_T("Conclusion : $\\lim_{n\\to+\\infty}\\int_I f_n\\,d\\lambda=\\int_I f\\,d\\lambda=" + _lx(J) + "$, établi par le calcul direct (aucun des deux théorèmes ne s'applique ici)."))
    elif egal is False:
        P(_T("Conclusion : la permutation est fausse pour cette suite ($" + _lx(L) + "\\ne" + _lx(J) + "$)."))

    # ---- graphes ----------------------------------------------------------------------------------------------
    try:
        _graphes_convergence(fn, n1, x, I, f_lim, g_maj, J, In, L, int(N))
    except Exception as e:  # pragma: no cover
        _rouge("Les graphes n'ont pas pu être produits : " + str(e))


def _graphes_convergence(fn, n1, x, I, f_lim, g_maj, J, In, L, N):
    _titre("Graphes")
    a, b = I.start, I.end
    wl, wh = _fenetre(a, b)
    xs = np.linspace(wl + (wh - wl) * 1e-6, wh - (wh - wl) * 1e-6, 3001)
    ns = sorted(set([1, 2, 3, 5, 10, max(2, N)]))
    ns = [k for k in ns if k <= max(N, 10)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    cmap = plt.get_cmap("viridis")
    ymax = 0.0
    for i, k in enumerate(ns):
        fk = _fnp(fn.subs(n1, k), x)
        if fk is None:
            continue
        ys = fk(xs)
        ys = np.where(np.isfinite(ys), ys, np.nan)
        ax.plot(xs, ys, color=cmap(i / max(1, len(ns) - 1)), lw=1.4, label="$f_{%d}$" % k)
        if np.isfinite(ys).any():
            ymax = max(ymax, float(np.nanpercentile(np.abs(ys), 98)))
    fl = _fnp(f_lim, x) if f_lim is not None else None
    if fl is not None:
        yl = fl(xs)
        yl = np.where(np.isfinite(yl), yl, np.nan)
        ax.plot(xs, yl, color="crimson", lw=2.2, ls="--", label="$f=\\lim f_n$")
    if g_maj is not None:
        gm = _fnp(g_maj, x)
        if gm is not None:
            yg = gm(xs)
            yg = np.where(np.isfinite(yg), yg, np.nan)
            ax.plot(xs, yg, color="k", lw=1.2, ls=":", label="majorant $g$")
            if np.isfinite(yg).any():
                ymax = max(ymax, float(np.nanpercentile(np.abs(yg), 90)))
    if ymax > 0:
        ax.set_ylim(-0.2 * ymax if ymax > 0 else None, 1.35 * ymax)
    ax.axhline(0, color="k", lw=0.7)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    _style_axes(ax, "Les fonctions $f_n$, leur limite simple" + (" et le majorant" if g_maj is not None else ""), "x", "y")
    st.pyplot(fig)

    # intégrales en fonction de n
    ks = list(range(1, max(N, 10) + 1))
    vals = []
    for k in ks:
        q = _quad_n(fn, n1, x, k, a, b)
        vals.append(q if q is not None else np.nan)
    if np.isfinite(np.array(vals, dtype=float)).any():
        fig2, ax2 = plt.subplots(figsize=(10, 3.8))
        ax2.plot(ks, vals, "o-", color="tab:blue", label="$\\int_I f_n$")
        if J is not None:
            try:
                ax2.axhline(float(sp.N(J)), color="crimson", ls="--", label="$\\int_I f$ (intégrale de la limite)")
            except Exception:
                pass
        ax2.legend(fontsize=8)
        _style_axes(ax2, "Les intégrales $\\int_I f_n$ face à $\\int_I \\lim f_n$", "n", "intégrale")
        st.pyplot(fig2)

    # distance L1 à la limite (si convergence dominée)
    if g_maj is not None and f_lim is not None:
        d = []
        for k in ks:
            try:
                dd = sp.Abs(fn.subs(n1, k) - f_lim)
                q = _quad(_fnum(dd, x), a, b)
            except Exception:
                q = None
            d.append(q if q is not None else np.nan)
        if np.isfinite(np.array(d, dtype=float)).any():
            fig3, ax3 = plt.subplots(figsize=(10, 3.6))
            ax3.semilogy(ks, np.maximum(np.array(d, dtype=float), 1e-16), "s-", color="tab:green")
            _style_axes(ax3, "Convergence dans $L^1$ : $\\int_I |f_n-f|\\to0$", "n", "$\\int |f_n-f|$")
            st.pyplot(fig3)


