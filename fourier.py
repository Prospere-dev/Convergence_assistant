# =============================================================
# fourier.py : étude des séries de Fourier (programme de L2)
#   - fonction T-périodique f, donnée sur une période [a, b[ (T = b - a)
#   - sorties via st.latex()/st.pyplot() comme le reste de l'application
# =============================================================
import io
import base64

import numpy as np
import sympy as sp
from sympy import S
from sympy.calculus.util import continuous_domain

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st = None  # renseigné par app.py : fourier.st = st

_n = sp.Symbol("n", integer=True, positive=True)


# -------------------------------------------------------------
# Outils généraux
# -------------------------------------------------------------
def _lat(e):
    return sp.latex(e)


def _txt(s):
    return r"\text{" + s + r"}"


def _titre(s):
    return r"\textbf{" + s + r"}"


def _num(e):
    """Valeur flottante d'une expression numérique, ou None."""
    try:
        v = complex(sp.N(e, 30))
        if abs(v.imag) < 1e-12:
            return v.real
        return None
    except Exception:
        return None


def _nul(e):
    """True si e est (numériquement ou formellement) nul."""
    try:
        e = sp.simplify(e)
        if e == 0:
            return True
        v = _num(e)
        return v is not None and abs(v) < 1e-12
    except Exception:
        return False


def _lim(expr, x, pt, sens):
    """Limite à droite ('+') ou à gauche ('-') en pt ; None si infinie ou indéfinie."""
    try:
        v = expr.subs(x, pt)
        if v.is_finite is True and not v.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
            return sp.simplify(v)
    except Exception:
        pass
    try:
        v = sp.limit(expr, x, pt, sens)
        if v.is_finite is True and not v.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
            return sp.simplify(v)
    except Exception:
        pass
    return None


# -------------------------------------------------------------
# Découpage de f en morceaux sur la période
# -------------------------------------------------------------
def _morceaux(f, x, a, b):
    """Liste triée de (expr, lo, hi) recouvrant ]a, b[ ; None si impossible."""
    f = sp.sympify(f)
    J = sp.Interval(a, b)
    res = []
    try:
        g = sp.piecewise_fold(f.rewrite(sp.Piecewise))
        if isinstance(g, sp.Piecewise):
            paires = g.as_expr_set_pairs(J)
        else:
            paires = [(g, J)]
        for e, ens in paires:
            parts = list(ens.args) if isinstance(ens, sp.Union) else [ens]
            for p in parts:
                if isinstance(p, sp.Interval) and p.start != p.end:
                    res.append((e, p.start, p.end))
    except Exception:
        res = _morceaux_abs(f, x, a, b)
        if res is None:
            return None
    res.sort(key=lambda t: (_num(t[1]) if _num(t[1]) is not None else 0.0))
    # contrôle de recouvrement
    if not res:
        return None
    if not _nul(res[0][1] - a) or not _nul(res[-1][2] - b):
        return None
    for i in range(len(res) - 1):
        if not _nul(res[i][2] - res[i + 1][1]):
            return None
    return res


def _morceaux_abs(f, x, a, b):
    """Repli : découpe aux zéros des arguments de |.| (cas |sin x|, |cos x|...)."""
    atomes = list(f.atoms(sp.Abs))
    coupes = [a, b]
    try:
        for A in atomes:
            z = sp.solveset(A.args[0], x, sp.Interval(a, b))
            if isinstance(z, sp.FiniteSet):
                coupes += [sp.simplify(t) for t in z.args]
            else:
                return None
    except Exception:
        return None
    uniques = []
    for c in sorted(coupes, key=lambda t: _num(t)):
        if not uniques or not _nul(c - uniques[-1]):
            uniques.append(c)
    res = []
    for lo, hi in zip(uniques[:-1], uniques[1:]):
        mil = (lo + hi) / 2
        rempl = {}
        for A in atomes:
            v = _num(A.args[0].subs(x, mil))
            if v is None:
                return None
            rempl[A] = A.args[0] if v >= 0 else -A.args[0]
        res.append((f.xreplace(rempl), lo, hi))
    return res


def _evalue_morceaux(pieces, x, xs):
    """Évalue f (numpy) sur un tableau xs pris dans [a, b]."""
    xs = np.asarray(xs, dtype=float)
    out = np.full(xs.shape, np.nan)
    fonctions = []
    for e, lo, hi in pieces:
        try:
            fn = sp.lambdify(x, e, modules=["numpy"])
        except Exception:
            fn = None
        fonctions.append((fn, float(sp.N(lo)), float(sp.N(hi))))
    for i, (fn, lo, hi) in enumerate(fonctions):
        if fn is None:
            continue
        if i == 0:
            masque = (xs >= lo) & (xs <= hi)
        else:
            masque = (xs > lo) & (xs <= hi)
        if not masque.any():
            continue
        with np.errstate(all="ignore"):
            v = fn(xs[masque])
        out[masque] = np.broadcast_to(np.asarray(v, dtype=float), xs[masque].shape)
    return out



# -------------------------------------------------------------
# Régularité, sauts, parité
# -------------------------------------------------------------
def _regularite(pieces, x):
    """(cpm, c1pm) : f continue par morceaux / de classe C^1 par morceaux sur la période."""
    cpm, c1 = True, True
    for e, lo, hi in pieces:
        I = sp.Interval.open(lo, hi)
        try:
            if continuous_domain(e, x, I) != I:
                return False, False
        except Exception:
            return False, False
        if _lim(e, x, lo, "+") is None or _lim(e, x, hi, "-") is None:
            return False, False
        if c1:
            try:
                d = sp.diff(e, x)
                if continuous_domain(d, x, I) != I or _lim(d, x, lo, "+") is None or _lim(d, x, hi, "-") is None:
                    c1 = False
            except Exception:
                c1 = False
    return cpm, c1


def _sauts(pieces, x):
    """Renvoie (sauts_intérieurs, raccord). Chaque saut : (x0, f(x0-), f(x0+))."""
    sauts = []
    for i in range(len(pieces) - 1):
        e1, _, hi = pieces[i]
        e2, lo2, _ = pieces[i + 1]
        g = _lim(e1, x, hi, "-")
        d = _lim(e2, x, lo2, "+")
        if g is None or d is None:
            continue
        if not _nul(d - g):
            sauts.append((hi, g, d))
    e_first, a, _ = pieces[0]
    e_last, _, b = pieces[-1]
    fa = _lim(e_first, x, a, "+")
    fb = _lim(e_last, x, b, "-")
    raccord = (a, b, fa, fb, (fa is not None and fb is not None and _nul(fa - fb)))
    return sauts, raccord


def _parite(pieces, x, a, b):
    """'paire', 'impaire' ou None (période centrée en 0)."""
    if not _nul(a + b):
        return None
    pts = [float(sp.N(b)) * t for t in (0.137, 0.291, 0.442, 0.573, 0.718, 0.861, 0.953)]
    vp = _evalue_morceaux(pieces, x, pts)
    vm = _evalue_morceaux(pieces, x, [-t for t in pts])
    if np.any(np.isnan(vp)) or np.any(np.isnan(vm)):
        return None
    if np.allclose(vp, vm, atol=1e-9):
        return "paire"
    if np.allclose(vp, -vm, atol=1e-9):
        return "impaire"
    return None


# -------------------------------------------------------------
# Coefficients de Fourier : calcul exact (avec contrôle numérique)
# -------------------------------------------------------------
def _generique(r, n):
    """Si r est un Piecewise en n, renvoie la branche valable pour n grand."""
    if isinstance(r, sp.Piecewise):
        for e, c in r.args:
            try:
                if bool(c.subs(n, 100003)):
                    return e
            except Exception:
                continue
        return None
    return r


def _coef_numeriques(pieces, x, w, T, K):
    """a_k, b_k (k=0..K) par quadrature de Gauss-Legendre composite."""
    xg, wg = np.polynomial.legendre.leggauss(24)
    noeuds, poids, fvals = [], [], []
    for e, lo, hi in pieces:
        lo_f, hi_f = float(sp.N(lo)), float(sp.N(hi))
        m = int(max(8, np.ceil(4 * K * (hi_f - lo_f) / float(sp.N(T))) + 4))
        bords = np.linspace(lo_f, hi_f, m + 1)
        fn = sp.lambdify(x, e, modules=["numpy"])
        for u, v in zip(bords[:-1], bords[1:]):
            t = 0.5 * (v - u) * xg + 0.5 * (u + v)
            with np.errstate(all="ignore"):
                val = np.broadcast_to(np.asarray(fn(t), dtype=float), t.shape)
            noeuds.append(t)
            poids.append(0.5 * (v - u) * wg)
            fvals.append(val)
    t = np.concatenate(noeuds)
    p = np.concatenate(poids)
    fv = np.concatenate(fvals)
    ks = np.arange(0, K + 1)[:, None]
    ang = ks * float(sp.N(w)) * t[None, :]
    A = (2.0 / float(sp.N(T))) * (np.cos(ang) * (p * fv)[None, :]).sum(axis=1)
    B = (2.0 / float(sp.N(T))) * (np.sin(ang) * (p * fv)[None, :]).sum(axis=1)
    return A, B


def _coefficients_calc(pieces, x, a, b, T, w, parite):
    """Calcule a_0, a_n, b_n (formules exactes si possible).
    Renvoie un dictionnaire : exact, a0, an, bn, an_exc, bn_exc, A, B (tableaux numériques)."""
    n = _n
    res = dict(exact=False, a0=None, an=None, bn=None, an_exc={}, bn_exc={}, A=None, B=None,
               formule_a=None, formule_b=None, pieces_int=pieces)
    # intégration sur la demi-période si parité connue (théorème sur la parité)
    pcs = pieces
    facteur = 2 / T
    if parite is not None:
        demi = []
        for e, lo, hi in pieces:
            if _num(hi) is not None and _num(hi) > 0:
                lo2 = lo if _num(lo) >= 0 else sp.Integer(0)
                demi.append((e, lo2, hi))
        pcs = demi
        facteur = 4 / T
    res["facteur"] = facteur
    res["pieces_int"] = pcs
    K = 400
    A_num, B_num = _coef_numeriques(pieces, x, w, T, K)
    res["A"], res["B"] = A_num, B_num
    try:
        def integr(g):
            lst = []
            for e, lo, hi in pcs:
                r = sp.integrate(e * g, (x, lo, hi))
                if r.has(sp.Integral):
                    return None
                lst.append(r)
            return lst

        # a_0
        if parite == "impaire":
            a0 = sp.Integer(0)
        else:
            lst0 = integr(sp.Integer(1))
            if lst0 is None:
                return res
            a0 = sp.simplify(facteur * sum(lst0))
        res["a0"] = a0
        for nom, g in (("an", sp.cos(n * w * x)), ("bn", sp.sin(n * w * x))):
            if (nom == "bn" and parite == "paire") or (nom == "an" and parite == "impaire"):
                res[nom] = sp.Integer(0)
                continue
            lst = integr(g)
            if lst is None:
                return res
            gens = [_generique(r, n) for r in lst]
            if any(gg is None for gg in gens):
                return res
            gen = sp.simplify(facteur * sum(gens))
            gen = _simplifie_n(gen, n)
            exc = {}
            for i in range(1, 9):
                try:
                    vals = []
                    for r in lst:
                        v = r.subs(n, i)
                        if v.has(sp.nan, sp.zoo, sp.oo, -sp.oo) or isinstance(v, sp.Piecewise):
                            v = sum(sp.integrate((e * g.subs(n, i)), (x, lo, hi)) for e, lo, hi in pcs)
                            vals = None
                            break
                        vals.append(v)
                    if vals is None:
                        v_i = sp.simplify(facteur * sum(sp.integrate((e * g.subs(n, i)), (x, lo, hi)) for e, lo, hi in pcs))
                    else:
                        v_i = sp.simplify(facteur * sum(vals))
                except Exception:
                    continue
                gv = gen.subs(n, i)
                if gv.has(sp.nan, sp.zoo, sp.oo, -sp.oo) or not _nul(gv - v_i):
                    exc[i] = v_i
            res[nom] = gen
            res[nom + "_exc"] = exc
        # contrôle numérique
        for nom, tab in (("an", A_num), ("bn", B_num)):
            gen = res[nom]
            for i in (1, 2, 3, 4, 5, 7, 12, 31):
                v = res[nom + "_exc"].get(i, None)
                if v is None:
                    v = gen.subs(n, i) if gen != 0 else 0
                vn = _num(v)
                if vn is None or abs(vn - tab[i]) > 2e-6 * (1 + abs(tab[i])):
                    return res
        a0n = _num(a0)
        if a0n is None or abs(a0n - A_num[0]) > 2e-6 * (1 + abs(A_num[0])):
            return res
        res["exact"] = True
    except Exception:
        res["exact"] = False
    return res


def _simplifie_n(e, n):
    try:
        e = sp.simplify(e)
        e = e.subs({sp.sin(sp.pi * n): 0})
        return sp.simplify(e)
    except Exception:
        return e


def _valeur(coef, nom, k):
    """a_k ou b_k exacts (k entier >= 1)."""
    exc = coef[nom + "_exc"]
    if k in exc:
        return exc[k]
    g = coef[nom]
    if g is None:
        return None
    return sp.simplify(g.subs(_n, k)) if g != 0 else sp.Integer(0)


def _coefficients(pieces, x, a, b, T, w, parite):
    """Calcul exact des coefficients avec délai maximal ; repli numérique sinon."""
    r = _avec_delai(lambda: _coefficients_calc(pieces, x, a, b, T, w, parite), 45)
    if r is not None:
        return r
    A, B = _coef_numeriques(pieces, x, w, T, 400)
    return dict(exact=False, a0=None, an=None, bn=None, an_exc={}, bn_exc={}, A=A, B=B, pieces_int=pieces, facteur=None)



# -------------------------------------------------------------
# Aides de rédaction
# -------------------------------------------------------------
_m = sp.Symbol("m", integer=True, nonnegative=True)


def _a_pi(e):
    """Écriture lisible d'un nombre exact."""
    try:
        return sp.latex(sp.nsimplify(e) if e.is_Float else sp.simplify(e))
    except Exception:
        return sp.latex(e)


def _cas_parite(expr, n):
    """Si expr (en n) dépend de la parité de n, renvoie le LaTeX par cas, sinon None."""
    try:
        depend = any(p.base == -1 and p.exp.has(n) for p in expr.atoms(sp.Pow)) or expr.has(sp.sin, sp.cos)
        if not depend:
            return None
        m = _m
        pair = sp.simplify(expr.subs(n, 2 * m))
        impair = sp.simplify(expr.subs(n, 2 * m + 1))
        pair_n = sp.simplify(pair.subs(m, n / 2))
        impair_n = sp.simplify(impair.subs(m, (n - 1) / 2))
        if pair_n.has(sp.sin, sp.cos) or impair_n.has(sp.sin, sp.cos):
            return None
        if sp.simplify(pair_n - impair_n) == 0:
            return None
        return (r"\begin{cases}" + sp.latex(pair_n) + r" & \text{si } n \text{ est pair}\\ "
                + sp.latex(impair_n) + r" & \text{si } n \text{ est impair}\end{cases}")
    except Exception:
        return None


def _formule_coef(nom, coef, n):
    """Lignes LaTeX donnant a_n ou b_n (formule générale, cas particuliers, cas de parité)."""
    g = coef[nom]
    exc = coef[nom + "_exc"]
    sym = "a_n" if nom == "an" else "b_n"
    lignes = []
    if g == 0 and not exc:
        lignes.append(sym + r"=0\quad(n\ge 1)")
        return lignes
    pref = sym + "=" + sp.latex(g)
    if exc:
        ens = ",\\ ".join(str(k) for k in sorted(exc))
        lignes.append(pref + r"\quad(n\ge 1,\ n\notin\{" + ens + r"\})")
        for k in sorted(exc):
            lignes.append(sym.replace("n", str(k)) + "=" + sp.latex(exc[k]))
    else:
        lignes.append(pref + r"\quad(n\ge 1)")
    cas = _cas_parite(g, n) if not exc else None
    if cas:
        lignes.append(sym + "=" + cas)
    return lignes


def _classes(terme, n):
    """Réindexation selon la parité de n : renvoie (expr(m), m0, texte) ou None.
    Utilisée pour écrire les sommes de séries numériques sans termes nuls."""
    m = _m
    try:
        pair = sp.simplify(terme.subs(n, 2 * m))
        impair = sp.simplify(terme.subs(n, 2 * m + 1))
    except Exception:
        return None
    if pair == 0 and impair != 0:
        return impair, 0, r"n=2m+1"
    if impair == 0 and pair != 0:
        return pair, 1, r"n=2m"
    return None


def _decompose_constante(terme, var):
    c, g = terme.as_independent(var, as_Add=False)
    if c == 0:
        return sp.Integer(1), terme
    return c, g


def _par(e):
    """LaTeX d'un nombre, entre parenthèses s'il est négatif."""
    s = sp.latex(e)
    try:
        if _num(e) is not None and _num(e) < 0:
            return r"\left(" + s + r"\right)"
    except Exception:
        pass
    return s



# -------------------------------------------------------------
# Calculs d'appui pour la rédaction
# -------------------------------------------------------------
def _avec_delai(fn, secondes=12):
    """Exécute fn() avec un délai maximal (fonctionne aussi hors du thread principal).
    Renvoie None en cas d'échec ou de dépassement du délai."""
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


def _noeuds(pieces, x, m=64):
    xg, wg = np.polynomial.legendre.leggauss(16)
    T_, P_, F_ = [], [], []
    for e, lo, hi in pieces:
        lo_f, hi_f = float(sp.N(lo)), float(sp.N(hi))
        fn = sp.lambdify(x, e, modules=["numpy"])
        bords = np.linspace(lo_f, hi_f, m + 1)
        for u, v in zip(bords[:-1], bords[1:]):
            t = 0.5 * (v - u) * xg + 0.5 * (u + v)
            with np.errstate(all="ignore"):
                val = np.broadcast_to(np.asarray(fn(t), dtype=float), t.shape)
            T_.append(t)
            P_.append(0.5 * (v - u) * wg)
            F_.append(val)
    return np.concatenate(T_), np.concatenate(P_), np.concatenate(F_)


def _moyenne_carre(pieces, x, T):
    """(1/T) * integrale de f^2 sur la période : (valeur exacte ou None, valeur flottante)."""
    exact = None
    try:
        def calc():
            tot = 0
            for e, lo, hi in pieces:
                r = sp.integrate(sp.expand(e ** 2), (x, lo, hi))
                if r.has(sp.Integral):
                    return None
                tot += r
            return sp.simplify(tot / T)
        exact = _avec_delai(calc, 15)
    except Exception:
        exact = None
    t, p, fv = _noeuds(pieces, x)
    num = float((p * fv ** 2).sum() / float(sp.N(T)))
    if exact is not None:
        v = _num(exact)
        if v is None or abs(v - num) > 1e-6 * (1 + abs(num)):
            exact = None
    return exact, num


def _ordre_decroissance(coef, n):
    """Ordre p tel que |a_n|+|b_n| ~ C/n^p. Renvoie (p, C, texte_classe) ou (None, None, None)."""
    m = sp.Symbol("m", integer=True, positive=True)
    if coef["exact"] and not coef["an_exc"] and not coef["bn_exc"]:
        M = sp.Abs(coef["an"]) + sp.Abs(coef["bn"])
        classes = []
        for r in (0, 1):
            Mr = sp.simplify(M.subs(n, 2 * m + r))
            classes.append(Mr)
        for p in range(0, 9):
            vals, ok = [], True
            for Mr in classes:
                if Mr == 0:
                    vals.append(sp.Integer(0))
                    continue
                l = _avec_delai(lambda: sp.limit(sp.simplify(Mr * m ** p), m, sp.oo), 8)
                if l is None or l.has(sp.oo, -sp.oo, sp.nan, sp.zoo):
                    ok = False
                    break
                vals.append(sp.simplify(l * 2 ** p))
            if ok and any(v != 0 for v in vals):
                if vals[0] == 0:
                    return p, vals[1], "impair"
                if vals[1] == 0:
                    return p, vals[0], "pair"
                return p, sp.Max(vals[0], vals[1]) if vals[0] != vals[1] else vals[0], None
        return None, None, None
    A, B = coef["A"], coef["B"]
    if A is None:
        return None, None, None
    M = np.abs(A[1:]) + np.abs(B[1:])
    ks = np.arange(1, len(M) + 1)
    def env(lo, hi):
        sel = (ks >= lo) & (ks < hi)
        return M[sel].max(), ks[sel].mean()
    m1, k1 = env(50, 100)
    m2, k2 = env(200, 400)
    if m1 <= 1e-14 or m2 <= 1e-14:
        return None, None, None
    p = -np.log(m2 / m1) / np.log(k2 / k1)
    pr = int(round(p))
    if abs(p - pr) < 0.2 and pr >= 0:
        return pr, sp.Float(m2 * k2 ** pr, 4), None
    return None, None, None



def _limites_en(pieces, x, a, b, T, x0):
    """Limites à gauche et à droite en x0 du prolongement T-périodique de f."""
    k = sp.floor(sp.simplify((x0 - a) / T))
    t = sp.simplify(x0 - k * T)
    if _nul(t - a):
        g = _lim(pieces[-1][0], x, b, "-")
        d = _lim(pieces[0][0], x, a, "+")
        return g, d
    for i in range(len(pieces) - 1):
        if _nul(t - pieces[i][2]):
            return _lim(pieces[i][0], x, pieces[i][2], "-"), _lim(pieces[i + 1][0], x, pieces[i + 1][1], "+")
    for e, lo, hi in pieces:
        if _num(t) is not None and _num(lo) < _num(t) < _num(hi):
            v = sp.simplify(e.subs(x, t))
            return v, v
    return None, None


def _somme_numerique(terme, m0, mmax=20000):
    """Somme partielle numérique de sum_{m>=m0} terme(m)."""
    try:
        fn = sp.lambdify(_m, terme, modules=["numpy"])
        ms = np.arange(m0, mmax + 1, dtype=float)
        with np.errstate(all="ignore"):
            v = np.broadcast_to(np.asarray(fn(ms), dtype=float), ms.shape)
        return float(np.sum(v))
    except Exception:
        return None


class _Lignes(list):
    cle = None


def _ecrit_somme(terme_n, n, valeur, intro=""):
    """Met en forme 'somme de terme_n pour n>=1 = valeur' avec réindexation et facteur constant.
    Renvoie une liste de lignes LaTeX, ou [] si la vérification numérique échoue."""
    m = _m
    terme_n = sp.simplify(terme_n)
    if terme_n == 0:
        return []
    cl = _classes(terme_n, n)
    if cl is not None:
        terme, m0, rel = cl
    else:
        terme, m0, rel = terme_n.subs(n, m), 1, None
        if terme_n.has(m):
            return []
    if terme.has(sp.sin, sp.cos):
        return []
    c, g = _decompose_constante(terme, m)
    c = sp.simplify(c)
    if c == 0:
        return []
    cible = sp.simplify(valeur / c)
    nv = _num(cible)
    ns = _somme_numerique(g, m0)
    if nv is None or ns is None or abs(ns - nv) > 5e-3 * (1 + abs(nv)):
        return []
    lat_g = sp.latex(g)
    # réécriture en variable n
    gn = g.subs(m, n)
    s_g = r"\sum_{n\ge " + str(m0) + r"}" + sp.latex(gn) if rel is None else r"\sum_{m\ge " + str(m0) + r"}" + lat_g
    pre = (r"\text{(termes d'indice } " + rel + r"\text{ seulement)}\quad ") if rel is not None else ""
    res = _Lignes([pre + s_g + "=" + sp.latex(cible)])
    try:
        res.cle = (round(abs(nv), 9), str(sp.simplify(sp.Abs(g))))
    except Exception:
        res.cle = None
    return res


def _ordre_texte(p, C, classe):
    if p is None:
        return None
    nn = sp.Symbol("n")
    corps = sp.latex(sp.simplify(C / nn ** p)) if p > 0 else sp.latex(C)
    sup = ""
    if classe:
        sup = r"\quad(n \text{ " + classe + r"})"
    return r"|a_n|+|b_n|\ \underset{n\to+\infty}{\sim}\ " + corps + sup



# =============================================================
# Étude complète d'une série de Fourier
# =============================================================
def _rouge(msg):
    safe = (msg or "").replace("\\", r"\textbackslash ").replace("_", r"\_")
    return r"\textcolor{red}{\text{" + safe + r"}}"


def etude_serie_de_fourier(f_expr, x, intervalle, N=10):
    P = st.latex
    n = _n
    _PRIM_VUES.clear()
    pi = sp.pi

    # ---- période ------------------------------------------------
    note_defaut = False
    if intervalle == S.Reals or not isinstance(intervalle, sp.Interval) or intervalle.start.is_infinite or intervalle.end.is_infinite:
        a, b = -pi, pi
        note_defaut = True
    else:
        a, b = intervalle.start, intervalle.end
    T = sp.simplify(b - a)
    w = sp.simplify(2 * pi / T)
    try:
        N = int(N)
    except Exception:
        N = 10
    N = max(1, min(N, 60))

    pieces = _morceaux(f_expr, x, a, b)
    if pieces is None:
        P(_rouge("Impossible de découper f en morceaux réguliers sur la période choisie."))
        return

    P(_titre("Données"))
    f_lat = sp.latex(sp.sympify(f_expr))
    P(r"\text{Fonction étudiée : } f(x)=" + f_lat + r"\quad \text{sur la période } [" + sp.latex(a) + r",\," + sp.latex(b) + r"[")
    if note_defaut:
        P(r"\text{Aucun intervalle fini n'est donné : la période choisie par défaut est } [-\pi,\pi[.")
    P(r"T=b-a=" + sp.latex(T) + r"\qquad \omega=\frac{2\pi}{T}=" + sp.latex(w))
    P(r"\text{On note } \tilde f \text{ le prolongement } T\text{-périodique de } f \text{ à } \mathbb{R} : \ \tilde f(x+T)=\tilde f(x).")

    # ---- régularité ---------------------------------------------
    P(_titre("Régularité de f"))
    rg = _avec_delai(lambda: _regularite(pieces, x), 30)
    cpm, c1 = rg if rg is not None else (False, False)
    if not cpm:
        P(_rouge("f n'est pas continue par morceaux sur la période (limite infinie ou non définie en un point)."))
        P(r"\text{Les théorèmes de Dirichlet et de Parseval du programme ne s'appliquent pas : l'étude s'arrête ici.}")
        return
    sauts, raccord = _sauts(pieces, x)
    a_, b_, fa, fb, rac_ok = raccord
    P(r"f \text{ est continue par morceaux sur } [" + sp.latex(a) + r"," + sp.latex(b) + r"] \text{ (limites finies à gauche et à droite en chaque point de raccord).}")
    if c1:
        P(r"f \text{ est de classe } C^1 \text{ par morceaux : sur chaque morceau } f \text{ et } f' \text{ se prolongent par continuité aux extrémités.}")
    else:
        P(r"f \text{ n'est pas de classe } C^1 \text{ par morceaux (dérivée non bornée au voisinage d'un point).}")
    if len(pieces) > 1:
        P(r"\text{Points de découpage : } " + ",\\ ".join(sp.latex(p[2]) for p in pieces[:-1]))
    disc = []
    for x0, g, d in sauts:
        disc.append(x0)
        P(r"\text{Discontinuité en } x=" + sp.latex(x0) + r"\text{ : } f(x^-)=" + sp.latex(g) + r",\ f(x^+)=" + sp.latex(d)
          + r",\ \text{saut } \delta=" + sp.latex(sp.simplify(d - g)))
    if fa is not None and fb is not None:
        if rac_ok:
            P(r"\text{Raccord des bords : } f(" + sp.latex(a) + r"^+)=f(" + sp.latex(b) + r"^-)=" + sp.latex(fa) + r"\text{ : } \tilde f \text{ est continue au raccord.}")
        else:
            disc.append(a)
            P(r"\text{Raccord des bords : } f(" + sp.latex(a) + r"^+)=" + sp.latex(fa) + r"\neq f(" + sp.latex(b) + r"^-)=" + sp.latex(fb)
              + r"\text{ : } \tilde f \text{ présente un saut } \delta=" + sp.latex(sp.simplify(fa - fb)) + r" \text{ en } x=" + sp.latex(a) + r"+kT.")
    continue_per = (len(sauts) == 0) and bool(rac_ok)
    if continue_per:
        P(r"\tilde f \text{ est continue sur } \mathbb{R}.")
    else:
        P(r"\tilde f \text{ n'est pas continue sur } \mathbb{R}.")

    # ---- parité -------------------------------------------------
    par = _parite(pieces, x, a, b)
    if par is not None:
        P(_titre("Parité"))
        if par == "paire":
            P(r"f \text{ est paire. Donc } b_n=0 \text{ pour tout } n \text{ et } a_n=\frac{4}{T}\int_0^{T/2} f(x)\cos(n\omega x)\,dx \quad(\text{série en cosinus}).")
        else:
            P(r"f \text{ est impaire. Donc } a_n=0 \text{ pour tout } n \text{ et } b_n=\frac{4}{T}\int_0^{T/2} f(x)\sin(n\omega x)\,dx \quad(\text{série en sinus}).")
        if len(pieces) == 1:
            fm = sp.simplify(pieces[0][0].subs(x, -x))
            P(r"f(-x)=" + sp.latex(fm) + r"=" + (r"-" if par == "impaire" else "") + r"f(x)\quad\text{donc } \tilde f \text{ est " + par + r" sur } [" + sp.latex(a) + r"," + sp.latex(b) + r"].")

    # ---- coefficients -------------------------------------------
    coef = _coefficients(pieces, x, a, b, T, w, par)
    P(_titre("Coefficients de Fourier"))
    P(r"a_0=\frac{2}{T}\int_{" + sp.latex(a) + r"}^{" + sp.latex(b) + r"} f(x)\,dx,\quad a_n=\frac{2}{T}\int_{" + sp.latex(a) + r"}^{" + sp.latex(b)
      + r"} f(x)\cos(n\omega x)\,dx,\quad b_n=\frac{2}{T}\int_{" + sp.latex(a) + r"}^{" + sp.latex(b) + r"} f(x)\sin(n\omega x)\,dx")
    P(r"c_n=\frac{1}{T}\int_{" + sp.latex(a) + r"}^{" + sp.latex(b) + r"} f(x)e^{-in\omega x}\,dx\quad(n\in\mathbb{Z}),\qquad c_0=\frac{a_0}{2},\ \ c_n=\frac{a_n-ib_n}{2},\ \ c_{-n}=\overline{c_n}\ \ (n\ge 1)")
    P(r"\text{(Les intégrales sur une période ne dépendent pas de la position de la période, car l'intégrande est } T\text{-périodique.)}")
    exact = coef["exact"]
    if exact:
        P(r"\text{Calcul des intégrales, } n\ge 1 \text{ entier :}")
        if len(pieces) > 1:
            P(r"f \text{ est définie par morceaux : } " + r" ;\ ".join(r"f(x)=" + sp.latex(e) + r"\text{ sur } [" + sp.latex(lo) + r"," + sp.latex(hi) + r"]" for e, lo, hi in pieces)
              + r"\text{. Chaque intégrale se découpe selon ces morceaux ; un morceau où } f=0 \text{ ne contribue pas.}")
        det0 = _detail_coef("a0", pieces, x, a, b, T, w, par, coef, f_lat)
        if det0:
            for l in det0:
                P(l)
        else:
            P(r"a_0=" + sp.latex(coef["a0"]))
        for nom in ("an", "bn"):
            det = _detail_coef(nom, pieces, x, a, b, T, w, par, coef, f_lat)
            fl = _formule_coef(nom, coef, n)
            if det:
                for l in det:
                    P(l)
                for l in fl[1:]:
                    P(l)
            else:
                P(fl[0])
                for k in sorted(coef[nom + "_exc"]):
                    dk = _detail_coef_k(nom, k, pieces, x, a, b, T, w, par, coef)
                    if dk:
                        P(r"\text{Cas particulier } n=" + str(k) + r"\text{ (calculé directement) :}")
                        for l in dk:
                            P(l)
                    else:
                        P(fl[1 + sorted(coef[nom + "_exc"]).index(k)] if len(fl) > 1 + sorted(coef[nom + "_exc"]).index(k) else "")
                for l in fl[1 + len(coef[nom + "_exc"]):]:
                    P(l)
        try:
            cn = sp.factor_terms(sp.expand((coef["an"] - sp.I * coef["bn"]) / 2))
            if not coef["an_exc"] and not coef["bn_exc"]:
                P(r"c_n=\frac{a_n-ib_n}{2}=" + sp.latex(cn) + r"\quad(n\ge 1),\qquad c_0=" + sp.latex(sp.simplify(coef["a0"] / 2)))
        except Exception:
            pass
    else:
        P(r"\text{Le calcul symbolique des intégrales n'aboutit pas : les coefficients sont calculés numériquement (quadrature de Gauss-Legendre).}")
        _nz = lambda v: sp.latex(sp.Integer(0)) if abs(v) < 1e-12 else sp.latex(sp.Float(v, 5))
        P(r"a_0\approx " + _nz(coef["A"][0]))
        P(r"a_n\ (n=1..5)\approx " + ",\\ ".join(_nz(v) for v in coef["A"][1:6]))
        P(r"b_n\ (n=1..5)\approx " + ",\\ ".join(_nz(v) for v in coef["B"][1:6]))

    # ---- série de Fourier ----------------------------------------
    P(_titre("Série de Fourier de f"))
    P(r"S(x)=\frac{a_0}{2}+\sum_{n\ge 1}\Big(a_n\cos(n\omega x)+b_n\sin(n\omega x)\Big)=\sum_{n\in\mathbb{Z}}c_n e^{in\omega x}")
    if exact:
        terme = None
        try:
            if not coef["an_exc"] and not coef["bn_exc"]:
                terme = sp.simplify(coef["an"] * sp.cos(n * w * x) + coef["bn"] * sp.sin(n * w * x))
        except Exception:
            terme = None
        if terme is not None:
            P(r"f(x)\ \sim\ " + sp.latex(sp.simplify(coef["a0"] / 2)) + r"+\sum_{n\ge 1}" + sp.latex(terme))
        else:
            premiers = []
            for k in range(1, 5):
                ak, bk = _valeur(coef, "an", k), _valeur(coef, "bn", k)
                premiers.append(sp.simplify(ak * sp.cos(k * w * x) + bk * sp.sin(k * w * x)))
            P(r"f(x)\ \sim\ " + sp.latex(sp.simplify(coef["a0"] / 2)) + r"+" + "+".join("(" + sp.latex(t) + ")" for t in premiers) + r"+\cdots")
    P(r"\text{Sommes partielles : } S_N(x)=\frac{a_0}{2}+\sum_{n=1}^{N}\big(a_n\cos(n\omega x)+b_n\sin(n\omega x)\big).")

    # ---- grandeurs communes pour Parseval ------------------------------
    Ef_exact, Ef_num = _moyenne_carre(pieces, x, T)
    A, B = coef["A"], coef["B"]
    ps = lambda K: A[0] ** 2 / 4 + 0.5 * float(np.sum(A[1:K + 1] ** 2 + B[1:K + 1] ** 2))
    ef_lat = sp.latex(Ef_exact) if Ef_exact is not None else sp.latex(sp.Float(Ef_num, 6))
    exact_ok = bool(exact and not coef["an_exc"] and not coef["bn_exc"])

    # ---- Dirichlet ---------------------------------------------------
    P(_titre("Convergence simple : théorème de Dirichlet appliqué à f"))
    if c1:
        P(r"\text{Vérification des hypothèses morceau par morceau (f et } f' \text{ doivent être continues avec des limites finies aux extrémités) :}")
        for e, lo, hi in pieces:
            try:
                de = sp.simplify(sp.diff(e, x))
                fl, fh = _lim(e, x, lo, "+"), _lim(e, x, hi, "-")
                dl, dh = _lim(de, x, lo, "+"), _lim(de, x, hi, "-")
                P(r"\text{Sur } ]" + sp.latex(lo) + r"," + sp.latex(hi) + r"[\text{ : } f(x)=" + sp.latex(e) + r",\ f'(x)=" + sp.latex(de)
                  + r"\ ;\quad f(" + sp.latex(lo) + r"^+)=" + sp.latex(fl) + r",\ f(" + sp.latex(hi) + r"^-)=" + sp.latex(fh)
                  + r",\ f'(" + sp.latex(lo) + r"^+)=" + sp.latex(dl) + r",\ f'(" + sp.latex(hi) + r"^-)=" + sp.latex(dh))
            except Exception:
                pass
        P(r"\text{Toutes ces limites sont finies : } f \text{ est } C^1 \text{ par morceaux. Le théorème de Dirichlet s'applique : la série de Fourier converge simplement sur } \mathbb{R} \text{ et sa somme est}")
        P(r"S(x)=\frac{\tilde f(x^+)+\tilde f(x^-)}{2}")
        # expression explicite de S sur [a, b[
        cas, vus, pts_raccord = [], set(), []
        pts = [a] + [p[2] for p in pieces[:-1]]
        for x0 in pts:
            g, d = _limites_en(pieces, x, a, b, T, x0)
            if g is None or d is None:
                continue
            val = sp.simplify((g + d) / 2)
            pts_raccord.append((x0, g, d, val))
        for e, lo, hi in pieces:
            cas.append(sp.latex(e) + r" & x\in\,]" + sp.latex(lo) + r"," + sp.latex(hi) + r"[")
        for x0, g, d, val in pts_raccord:
            cas.append(sp.latex(val) + r" & x=" + sp.latex(x0))
        P(r"\text{Ici, sur la période } [" + sp.latex(a) + r"," + sp.latex(b) + r"[\text{ :}\quad S(x)=\begin{cases}" + r"\\ ".join(cas) + r"\end{cases}\quad\text{puis } S \text{ est } T\text{-périodique.}")
        for x0, g, d, val in pts_raccord:
            if _nul(g - d):
                continue
            cle = sp.simplify(x0 - T * sp.floor(sp.simplify((x0 - a) / T)))
            if cle in vus:
                continue
            vus.add(cle)
            P(r"\text{En } x=" + sp.latex(x0) + r"+kT\ (k\in\mathbb{Z})\text{ : } \tilde f(x^-)=" + sp.latex(g) + r",\ \tilde f(x^+)=" + sp.latex(d)
              + r"\ \Longrightarrow\ S(x)=\frac{" + _par(g) + "+" + _par(d) + r"}{2}=" + sp.latex(val) + r"\ \ (\ne \tilde f(x^+) \text{ en général : la série ignore la valeur de } f \text{ en ce point}).")
        if continue_per:
            P(r"\tilde f \text{ est continue sur } \mathbb{R} : \ S=\tilde f \text{ partout.}")
    else:
        P(r"\text{Ici } f \text{ n'est pas de classe } C^1 \text{ par morceaux (dérivée non bornée) : Dirichlet ne s'applique pas ; seule la convergence en moyenne quadratique (Parseval) reste disponible.}")

    # ---- convergence normale / uniforme -------------------------------
    P(_titre("Convergence normale et convergence uniforme"))
    p_ord, C_ord, cl_ord = _ordre_decroissance(coef, n)
    t_ord = _ordre_texte(p_ord, C_ord, cl_ord)
    if exact_ok:
        try:
            Mn = sp.simplify(sp.sqrt(sp.simplify(coef["an"] ** 2 + coef["bn"] ** 2)))
            P(r"\sup_{x}\left|a_n\cos(n\omega x)+b_n\sin(n\omega x)\right|=\sqrt{a_n^2+b_n^2}=" + sp.latex(Mn)
              + r"\quad(n\ge 1),\qquad |a_n|+|b_n|=" + sp.latex(sp.simplify(sp.Abs(coef["an"]) + sp.Abs(coef["bn"]))))
        except Exception:
            pass
    if t_ord:
        P(t_ord)
    if c1 and continue_per:
        if p_ord is not None and p_ord > 1:
            P(r"\text{Exposant } " + str(p_ord) + r">1 : \ \sum(|a_n|+|b_n|) \text{ converge (série de Riemann), donc la série de Fourier converge normalement, donc uniformément, sur } \mathbb{R}, \text{ vers } S=\tilde f.")
        else:
            P(r"\tilde f \text{ est continue et } C^1 \text{ par morceaux : la série de Fourier converge normalement, donc uniformément, sur } \mathbb{R} \text{ vers } \tilde f \text{ (les coefficients de } f \text{ se déduisent de ceux de } f' \text{ par } a_n=-\tfrac{\beta_n}{n\omega},\ b_n=\tfrac{\alpha_n}{n\omega}\text{).}")
        P(r"\text{Conséquence : } S=\tilde f \text{ est continue ; on peut intégrer terme à terme (voir plus bas).}")
    elif c1:
        P(r"\text{Chaque somme partielle } S_N \text{ est continue. Si la convergence était uniforme sur } \mathbb{R}\text{, la somme } S \text{ serait continue.}")
        sj = [(x0, g, d) for x0, g, d, val in pts_raccord if not _nul(g - d)]
        if sj:
            x0, g, d = sj[0]
            P(r"\text{Or, d'après Dirichlet, } S=\tilde f \text{ hors des points de saut : en } x=" + sp.latex(x0) + r"\text{, } S(x^-)=" + sp.latex(g) + r"\ne S(x^+)=" + sp.latex(d)
              + r"\text{ : } S \text{ est discontinue.}")
        P(r"\text{Il n'y a donc ni convergence uniforme, ni convergence normale sur } \mathbb{R}.")
        if p_ord == 1:
            P(r"\text{Contrôle : } \sum(|a_n|+|b_n|)=\sum " + (sp.latex(sp.simplify(C_ord / n)) if C_ord is not None else r"\frac{C}{n}") + r"\text{ est une série harmonique (à constante près) : elle diverge.}")
        # segments sans discontinuité
        try:
            reps = []
            for x0 in disc:
                r_ = sp.simplify(x0 - T * sp.floor(sp.simplify((x0 - a) / T)))
                if not any(_nul(r_ - q) for q in reps):
                    reps.append(r_)
            reps.sort(key=lambda q: _num(q))
            segs = [r"]" + sp.latex(reps[i]) + r"," + sp.latex(reps[i + 1] if i + 1 < len(reps) else reps[0] + T) + r"["
                    for i in range(len(reps))]
            P(r"\text{Sur tout segment } [c,d]\subset " + r"\ ;\ ".join(segs) + r"\ (\text{et ses translatés de } kT)\text{, } \tilde f \text{ est continue et } C^1 \text{ et } S=\tilde f : \text{ la convergence y est uniforme.}")
        except Exception:
            pass
        try:
            gap = None
            for x0, g, d, val in pts_raccord:
                if not _nul(g - d):
                    gap = sp.simplify(d - g)
                    xg = x0
                    break
            if gap is not None:
                P(r"\text{Phénomène de Gibbs : en } x=" + sp.latex(xg) + r"+kT \text{ le saut vaut } \delta=" + sp.latex(gap)
                  + r"\text{ ; les sommes partielles } S_N \text{ dépassent la valeur limite d'environ } 8{,}95\%\text{ du saut, soit } \approx "
                  + sp.latex(sp.Float(0.08949 * abs(float(gap)), 3)) + r"\text{, quel que soit } N \text{ (voir les graphes).}")
        except Exception:
            pass
    else:
        P(r"\text{Sans la régularité } C^1 \text{ par morceaux, on ne peut pas conclure sur la convergence normale ou uniforme.}")

    # ---- Parseval -----------------------------------------------------
    P(_titre("Convergence en moyenne quadratique : égalité de Parseval"))
    P(r"\text{Parseval : } \frac1T\int_{" + sp.latex(a) + r"}^{" + sp.latex(b) + r"}|f(x)|^2dx=\frac{a_0^2}{4}+\frac12\sum_{n\ge 1}(a_n^2+b_n^2)")
    lE, Ev = _detail_energie(pieces, x, T) if exact else ([], None)
    if lE and Ef_exact is not None and _nul(Ev - Ef_exact):
        P(r"\text{Membre de gauche :}")
        for l in lE:
            P(l)
    else:
        P(r"\text{Membre de gauche : } \frac1T\int_{" + sp.latex(a) + r"}^{" + sp.latex(b) + r"} f^2\,dx=" + ef_lat)
    deja = set()
    dec_somme = []
    if exact_ok and Ef_exact is not None:
        try:
            Q = sp.simplify(coef["an"] ** 2 + coef["bn"] ** 2)
            W = sp.simplify(2 * (Ef_exact - coef["a0"] ** 2 / 4))
            P(r"\text{Membre de droite : } a_0=" + sp.latex(coef["a0"]) + r",\ a_n^2+b_n^2=" + sp.latex(Q) + r"\quad\text{donc}\quad "
              + sp.latex(sp.simplify(coef["a0"] ** 2 / 4)) + r"+\frac12\sum_{n\ge 1}" + _pq(Q))
            P(r"\text{Égalité : } " + sp.latex(sp.simplify(coef["a0"] ** 2 / 4)) + r"+\frac12\sum_{n\ge 1}" + _pq(Q) + r"=" + ef_lat
              + r"\ \Longrightarrow\ \sum_{n\ge 1}" + _pq(Q) + r"=" + sp.latex(W))
            dec_somme = _avec_delai(lambda: _ecrit_somme(Q, n, W), 20) or []
            if dec_somme:
                P(r"\text{Résultat numérique obtenu grâce à Parseval :}")
                for l in dec_somme:
                    deja.add(l)
                    P(l)
            for l in _reste_parseval(Q, n, N, Ef_exact, coef["a0"]):
                P(l)
        except Exception:
            pass
    else:
        reste = Ef_num - ps(N)
        P(r"\|f-S_{" + str(N) + r"}\|_2^2=\frac1T\int f^2-\Big(\frac{a_0^2}{4}+\frac12\sum_{n=1}^{" + str(N) + r"}(a_n^2+b_n^2)\Big)\approx " + sp.latex(sp.Float(max(reste, 0.0), 4)) + r"\text{ et tend vers } 0 \text{ quand } N\to+\infty \text{ (Parseval).}")

    # ---- applications de Dirichlet : sommes de séries -------------------
    if c1 and exact and not coef["an_exc"] and not coef["bn_exc"]:
        cand = [a, sp.simplify(a + T / 4), sp.simplify(a + T / 2), sp.simplify(a + 3 * T / 4)]
        if _num(a) <= 0 < _num(b):
            cand.append(sp.Integer(0))
        cand += [s[0] for s in sauts]
        blocs, vus = [], set()
        for x0 in cand:
            if len(blocs) >= 4:
                break
            g, d = _limites_en(pieces, x, a, b, T, x0)
            if g is None or d is None:
                continue
            S0 = sp.simplify((g + d) / 2)
            try:
                terme = sp.simplify(coef["an"] * sp.cos(n * w * x0) + coef["bn"] * sp.sin(n * w * x0))
            except Exception:
                continue
            if terme == 0 or sp.count_ops(terme) > 40:
                continue
            lignes = _avec_delai(lambda: _ecrit_somme(terme, n, S0 - coef["a0"] / 2), 20) or []
            cle = getattr(lignes, "cle", None) or "".join(lignes)
            if not lignes or cle in vus or all(l in deja for l in lignes):
                continue
            vus.add(cle)
            blocs.append((x0, S0, terme, lignes))
        if blocs:
            P(_titre("Applications : sommes de séries numériques (Dirichlet)"))
        for x0, S0, terme, lignes in blocs:
            g_, d_ = _limites_en(pieces, x, a, b, T, x0)
            if g_ is not None and d_ is not None and _nul(g_ - d_):
                P(r"\text{En } x=" + sp.latex(x0) + r"\text{ : } \tilde f \text{ est continue, donc } S(" + sp.latex(x0) + r")=\tilde f(" + sp.latex(x0) + r")=" + sp.latex(S0))
            else:
                P(r"\text{En } x=" + sp.latex(x0) + r"\text{ : } \tilde f(x^-)=" + sp.latex(g_) + r",\ \tilde f(x^+)=" + sp.latex(d_) + r"\Longrightarrow S(" + sp.latex(x0) + r")=" + sp.latex(S0))
            P(r"\text{On remplace } x \text{ par } " + sp.latex(x0) + r"\text{ dans la série : } \sum_{n\ge 1}\big(a_n\cos(n\omega x)+b_n\sin(n\omega x)\big)=S(" + sp.latex(x0) + r")-\frac{a_0}{2}="
              + sp.latex(sp.simplify(S0 - coef["a0"] / 2)))
            P(r"\text{Terme général en } x=" + sp.latex(x0) + r"\text{ : } a_n\cos(n\omega x)+b_n\sin(n\omega x)=" + sp.latex(terme))
            for l in lignes:
                P(l)

    # ---- dérivation et intégration terme à terme --------------------------
    P(_titre("Dérivation et intégration terme à terme"))
    deltas = [(s_[0], sp.simplify(s_[2] - s_[1])) for s_ in sauts]
    if not rac_ok and fa is not None and fb is not None:
        deltas.append((a, sp.simplify(fa - fb)))
    if c1:
        if exact_ok:
            try:
                corr_c = sum((dj * sp.cos(n * w * xj) for xj, dj in deltas), sp.Integer(0))
                corr_s = sum((dj * sp.sin(n * w * xj) for xj, dj in deltas), sp.Integer(0))
                al = sp.simplify(n * w * coef["bn"] - (2 / T) * corr_c)
                be = sp.simplify(-n * w * coef["an"] - (2 / T) * corr_s)
                a0p = sp.simplify((2 / T) * sum((_lim(e, x, hi, "-") - _lim(e, x, lo, "+")) for e, lo, hi in pieces))
                P(r"\text{Coefficients de Fourier de } f' \text{ (intégration par parties sur chaque morceau) :}")
                P(r"\alpha_0=\frac2T\int f'(x)\,dx=\frac2T\sum_{\text{morceaux}}\big(f(\text{fin}^-)-f(\text{début}^+)\big)=" + sp.latex(a0p))
                if deltas:
                    P(r"\alpha_n=n\omega b_n-\frac2T\sum_j\delta_j\cos(n\omega x_j)=" + sp.latex(sp.simplify(n * w * coef["bn"])) + r"-\left(" + sp.latex(sp.simplify((2 / T) * corr_c))
                      + r"\right)=" + sp.latex(al) + r",\qquad \text{avec } " + ",\\ ".join(r"x_j=" + sp.latex(xj) + r",\ \delta_j=" + sp.latex(dj) for xj, dj in deltas))
                    P(r"\beta_n=-n\omega a_n-\frac2T\sum_j\delta_j\sin(n\omega x_j)=" + sp.latex(sp.simplify(-n * w * coef["an"])) + r"-\left(" + sp.latex(sp.simplify((2 / T) * corr_s))
                      + r"\right)=" + sp.latex(be))
                else:
                    P(r"\alpha_n=n\omega b_n=" + sp.latex(al) + r",\qquad \beta_n=-n\omega a_n=" + sp.latex(be))
                if al == 0 and be == 0:
                    P(r"\text{Donc la série de Fourier de } f' \text{ se réduit à } \frac{\alpha_0}{2}=" + sp.latex(sp.simplify(a0p / 2))
                      + (r"\text{, ce qui est bien } f'(x)=" + sp.latex(sp.simplify(sp.diff(pieces[0][0], x))) if len(pieces) == 1 else "") + r".")
                amp = sp.simplify(n * w * sp.sqrt(sp.simplify(coef["an"] ** 2 + coef["bn"] ** 2)))
                if deltas:
                    P(r"\text{La série dérivée terme à terme } \sum n\omega\big(b_n\cos(n\omega x)-a_n\sin(n\omega x)\big) \text{ a un terme général d'amplitude } n\omega\sqrt{a_n^2+b_n^2}=" + sp.latex(amp)
                      + (r"\text{, qui ne tend pas vers } 0 \text{ : elle diverge grossièrement.}" if _nul(sp.limit(amp, n, sp.oo)) is False else r"\text{ (série dérivée terme à terme distincte de la série de } f' \text{ à cause des sauts).}"))
                else:
                    P(r"\text{La série de } f' \text{ est ici la série dérivée terme à terme de celle de } f.")
            except Exception:
                pass
        elif A is not None and deltas:
            w_f = float(sp.N(w))
            ks = np.arange(300, 401)
            amp_ = ks * w_f * np.sqrt(A[300:401] ** 2 + B[300:401] ** 2)
            if amp_.max() > 1e-3:
                P(r"\text{Le terme général de la série dérivée, d'amplitude } n\omega\sqrt{a_n^2+b_n^2}\approx " + sp.latex(sp.Float(float(amp_.mean()), 3))
                  + r"\text{ pour } n\approx 300\text{--}400, \text{ ne tend pas vers } 0 : \text{ la série dérivée diverge grossièrement.}")
    else:
        P(r"\text{Dérivation terme à terme : non étudiée (} f \text{ n'est pas } C^1 \text{ par morceaux).}")
    if exact_ok:
        try:
            tt = sp.Symbol("t", real=True)
            prec = sp.Integer(0)
            k0 = 0
            for k0, (e0, lo0, hi0) in enumerate(pieces):
                if sp.simplify(e0) != 0:
                    break
                prec += 0
            else:
                k0 = 0
            e0, lo0, hi0 = pieces[k0]
            avant = sum((sp.integrate(e_, (x, lo_, hi_)) for e_, lo_, hi_ in pieces[:k0]), sp.Integer(0))
            F0 = sp.simplify(avant + sp.integrate(e0.subs(x, tt), (tt, lo0, x)))
            if not F0.has(sp.Integral):
                te = sp.simplify(coef["an"] / (n * w) * (sp.sin(n * w * x) - sp.sin(n * w * a)) - coef["bn"] / (n * w) * (sp.cos(n * w * x) - sp.cos(n * w * a)))
                P(r"\text{Intégration terme à terme (toujours licite pour une série de Fourier). Pour } x\in[" + sp.latex(lo0) + r"," + sp.latex(hi0) + r"]\text{ :}")
                P(r"\int_{" + sp.latex(a) + r"}^{x} f(t)\,dt=" + sp.latex(F0))
                P(r"\int_{" + sp.latex(a) + r"}^{x} f(t)\,dt=\frac{a_0}{2}\left(" + sp.latex(x - a) + r"\right)+\sum_{n\ge 1}\left[\frac{a_n}{n\omega}\big(\sin(n\omega x)-\sin(n\omega a)\big)-\frac{b_n}{n\omega}\big(\cos(n\omega x)-\cos(n\omega a)\big)\right]")
                P(r"\text{Ici : } " + sp.latex(F0) + r"=" + sp.latex(sp.simplify(coef["a0"] / 2 * (x - a))) + r"+\sum_{n\ge 1}" + sp.latex(te))
        except Exception:
            pass

    # ---- graphes (toujours en dernier) --------------------------------------
    try:
        _graphes_fourier(pieces, x, a, b, T, w, coef, disc, sauts, rac_ok, Ef_num, N, c1, continue_per, p_ord)
    except Exception as e:  # pragma: no cover
        P(_rouge("Les graphes n'ont pas pu être produits : " + str(e)))

# -------------------------------------------------------------
# Rédaction détaillée des calculs (primitives, bornes, résultats)
# -------------------------------------------------------------
def _pq(Q):
    """LaTeX de Q, entre parenthèses si c'est une somme."""
    return r"\left(" + sp.latex(Q) + r"\right)" if Q.is_Add else sp.latex(Q)


def _methode_prim(g, x):
    """Courte description de la technique d'intégration utilisée pour g."""
    try:
        trig = [t for t in g.atoms(sp.sin, sp.cos, sp.exp) if t.has(x)]
        if not trig:
            return ""
        tmp = g.xreplace({t: 1 for t in trig})
        if len(trig) >= 2 and not tmp.has(x):
            return r"\text{ (produit de fonctions trigonométriques : linéarisation en somme)}"
        if tmp.is_polynomial(x):
            d = sp.degree(sp.expand(tmp), x) if tmp.has(x) else 0
            if d == 1:
                return r"\text{ (intégration par parties : } u=" + sp.latex(sp.expand(tmp)) + r"\text{, } v' \text{ = le facteur trigonométrique)}"
            if d >= 2:
                return r"\text{ (" + str(d) + r" intégrations par parties successives)}"
        return ""
    except Exception:
        return ""


_PRIM_VUES = set()


def _chaine_alignee(g, x, lo, hi, val):
    """Chaîne d'égalités alignées (parties, changement de variable, justifications) fournie par lebesgue.py ; None si indisponible."""
    try:
        import lebesgue as _LB
    except Exception:
        return None
    try:
        return _LB._chaine_int(g, x, lo, hi, False, False, val)
    except Exception:
        return None


def _int_detail(g, x, lo, hi, lignes, signe_id=None):
    """Ajoute à `lignes` : primitive de g, évaluation aux bornes ; renvoie la valeur de l'intégrale (ou None)."""
    F = _avec_delai(lambda: sp.integrate(g, x), 15)
    if F is None or F.has(sp.Integral):
        return None
    try:
        F = sp.simplify(F)
        val = sp.simplify(F.subs(x, hi) - F.subs(x, lo))
    except Exception:
        return None
    if val.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        return None
    bloc = _chaine_alignee(g, x, lo, hi, val)
    if bloc:
        lignes.append(bloc)
        return val
    ligne_prim = r"\text{Une primitive de } " + sp.latex(g) + r" \text{ est } F(x)=" + sp.latex(F) + _methode_prim(g, x)
    if ligne_prim not in _PRIM_VUES:
        _PRIM_VUES.add(ligne_prim)
        lignes.append(ligne_prim)
    lignes.append(r"\int_{" + sp.latex(lo) + r"}^{" + sp.latex(hi) + r"}" + sp.latex(g) + r"\,dx=\Big[F(x)\Big]_{" + sp.latex(lo) + r"}^{" + sp.latex(hi)
                  + r"}=F(" + sp.latex(hi) + r")-F(" + sp.latex(lo) + r")=" + sp.latex(val))
    return val


def _detail_coef(nom, pieces, x, a, b, T, w, par, coef, f_lat):
    """Lignes détaillées du calcul de a_0, a_n ou b_n (ou [] si le détail n'est pas disponible)."""
    n = _n
    sym = {"a0": "a_0", "an": "a_n", "bn": "b_n"}[nom]
    if not coef["exact"]:
        return []
    cible = {"a0": coef["a0"], "an": coef["an"], "bn": coef["bn"]}[nom]
    if cible is None:
        return []
    trig = {"a0": sp.Integer(1), "an": sp.cos(n * w * x), "bn": sp.sin(n * w * x)}[nom]
    lat_trig = "" if nom == "a0" else (r"\cdot " + sp.latex(trig))
    lignes = []
    # --- coefficient nul par parité
    if (par == "impaire" and nom in ("a0", "an")) or (par == "paire" and nom == "bn"):
        typ = "impaire" if par == "impaire" else "paire"
        autre = "paire" if typ == "impaire" else "impaire"
        if nom == "a0":
            lignes.append(sym + r"=\frac{2}{T}\int_{" + sp.latex(a) + r"}^{" + sp.latex(b) + r"}" + f_lat + r"\,dx=0 \quad\text{car } f \text{ est " + typ + r" et l'intervalle est symétrique.}")
        else:
            lignes.append(sym + r"=\frac{2}{T}\int_{" + sp.latex(a) + r"}^{" + sp.latex(b) + r"}" + f_lat + lat_trig + r"\,dx=0 \quad\text{car } x\mapsto "
                          + f_lat + lat_trig + r" \text{ est " + ("impaire" if par == "impaire" else "impaire") + r" (produit d'une fonction " + typ + r" et d'une fonction " + autre + r").}")
        return lignes
    # --- calcul par morceaux (demi-période si la parité le permet)
    demi = par is not None
    if demi:
        pcs = []
        for e, lo, hi in pieces:
            if _num(hi) is not None and _num(hi) > 1e-12:
                lo2 = lo if (_num(lo) is not None and _num(lo) > 0) else sp.Integer(0)
                pcs.append((e, lo2, hi))
        fact = sp.simplify(4 / T)
        lo_g, hi_g = sp.Integer(0), b
    else:
        pcs = list(pieces)
        fact = sp.simplify(2 / T)
        lo_g, hi_g = a, b
    if not pcs:
        return []
    enonce_f = f_lat if len(pcs) == 1 else r"f(x)"
    lignes.append(sym + r"=" + sp.latex(fact) + r"\int_{" + sp.latex(lo_g) + r"}^{" + sp.latex(hi_g) + r"}" + enonce_f + lat_trig + r"\,dx"
                  + (r"\quad(f \text{ est " + par + r" : on intègre sur } [0," + sp.latex(b) + r"] \text{ et on double)}" if demi else ""))
    total = 0
    for e, lo, hi in pcs:
        g = sp.expand(e * trig) if nom != "a0" else e
        if sp.simplify(e) == 0:
            continue
        v = _int_detail(g, x, lo, hi, lignes)
        if v is None:
            return []
        total += v
    res = sp.simplify(fact * total)
    try:
        d = sp.simplify(res - cible)
        if d != 0:
            ok = all(abs(complex(sp.N(d.subs(n, k)))) < 1e-9 for k in (2, 3, 4, 5, 7)) if d.has(n) else abs(complex(sp.N(d))) < 1e-9
            if not ok:
                return []
    except Exception:
        return []
    lignes.append(sym + r"=" + sp.latex(fact) + r"\times\left(" + sp.latex(sp.simplify(total)) + r"\right)=" + sp.latex(res)
                  + (r"\quad(n\ge 1)" if nom != "a0" else ""))
    return lignes


def _detail_energie(pieces, x, T):
    """Calcul explicite de (1/T) * integrale de f^2. Renvoie (lignes, valeur) ou ([], None)."""
    lignes, tot = [], 0
    for e, lo, hi in pieces:
        g = sp.expand(e ** 2)
        if sp.simplify(e) == 0:
            continue
        v = _int_detail(g, x, lo, hi, lignes)
        if v is None:
            return [], None
        tot += v
    val = sp.simplify(tot / T)
    lignes.append(r"\frac1T\int f^2\,dx=\frac{1}{" + sp.latex(T) + r"}\times\left(" + sp.latex(sp.simplify(tot)) + r"\right)=" + sp.latex(val))
    return lignes, val


def _reste_parseval(Q, n, N, Ef, a0):
    """Reste de Parseval ||f - S_N||^2 = 1/2 * somme_{n>N} Q(n), borne par une intégrale."""
    lignes = []
    t = sp.Symbol("t", positive=True)
    Nn = sp.Symbol("N", positive=True)
    try:
        vals = [float(sp.N(Q.subs(n, k))) for k in range(1, 61)]
        decroit = all(vals[i + 1] <= vals[i] + 1e-12 for i in range(len(vals) - 1))
    except Exception:
        return []
    lignes.append(r"\|f-S_N\|_2^2=\frac12\sum_{n>N}(a_n^2+b_n^2)=\frac12\sum_{n>N}" + _pq(Q))
    if not decroit:
        return lignes
    I = _avec_delai(lambda: sp.integrate(Q.subs(n, t), (t, Nn, sp.oo)), 10)
    if I is None or I.has(sp.Integral) or I.has(sp.oo, sp.nan, sp.zoo):
        return lignes
    borne = sp.simplify(I / 2)
    lignes.append(r"\text{La suite } n\mapsto " + sp.latex(Q) + r"\text{ est décroissante, donc }\ \|f-S_N\|_2^2\le \frac12\int_N^{+\infty}" + sp.latex(Q.subs(n, t))
                  + r"\,dt=" + sp.latex(borne))
    try:
        lignes.append(r"\text{Pour } N=" + str(int(N)) + r"\text{ : } \|f-S_{" + str(int(N)) + r"}\|_2^2\le " + sp.latex(sp.N(borne.subs(Nn, N), 4))
                      + r"\text{ ; cette borne tend vers } 0 \text{ quand } N\to+\infty\text{, donc } S_N\to f \text{ en moyenne quadratique.}")
    except Exception:
        pass
    return lignes


def _detail_coef_k(nom, k, pieces, x, a, b, T, w, par, coef):
    """Calcul explicite de a_k ou b_k pour une valeur particulière k de n (cas exceptionnels)."""
    sym = ("a" if nom == "an" else "b") + "_{" + str(k) + "}"
    cible = coef[nom + "_exc"].get(k)
    if cible is None:
        return []
    trig = sp.cos(k * w * x) if nom == "an" else sp.sin(k * w * x)
    demi = par is not None and ((par == "paire" and nom == "an") or (par == "impaire" and nom == "bn"))
    if demi:
        pcs = [(e, (lo if (_num(lo) is not None and _num(lo) > 0) else sp.Integer(0)), hi) for e, lo, hi in pieces if _num(hi) is not None and _num(hi) > 1e-12]
        fact, lo_g, hi_g = sp.simplify(4 / T), sp.Integer(0), b
    else:
        pcs, fact, lo_g, hi_g = list(pieces), sp.simplify(2 / T), a, b
    lignes = [sym + r"=" + sp.latex(fact) + r"\int_{" + sp.latex(lo_g) + r"}^{" + sp.latex(hi_g) + r"}f(x)\cdot " + sp.latex(trig) + r"\,dx"]
    tot = 0
    for e, lo, hi in pcs:
        if sp.simplify(e) == 0:
            continue
        v = _int_detail(sp.expand(e * trig), x, lo, hi, lignes)
        if v is None:
            return []
        tot += v
    res = sp.simplify(fact * tot)
    if abs(complex(sp.N(res - cible))) > 1e-9:
        return []
    lignes.append(sym + r"=" + sp.latex(fact) + r"\times\left(" + sp.latex(sp.simplify(tot)) + r"\right)=" + sp.latex(res)
                  + r"\quad(\text{cas particulier : le dénominateur } n^2-" + str(k) + r"^2 \text{ de la formule générale s'annulerait})" if False else
                  sym + r"=" + sp.latex(fact) + r"\times\left(" + sp.latex(sp.simplify(tot)) + r"\right)=" + sp.latex(res))
    return lignes



# =============================================================
# Graphes
# =============================================================
def _style_axes(ax, titre=None, xl=None, yl=None):
    ax.grid(True, alpha=0.25)
    if titre:
        ax.set_title(titre, fontsize=11)
    if xl:
        ax.set_xlabel(xl)
    if yl:
        ax.set_ylabel(yl)


def _graphes_fourier(pieces, x, a, b, T, w, coef, disc, sauts, rac_ok, Ef_num, N, c1, continue_per, p_ord):
    P = st.latex
    af, bf, Tf, wf = float(sp.N(a)), float(sp.N(b)), float(sp.N(T)), float(sp.N(w))
    A, B = coef["A"], coef["B"]
    K = len(A) - 1
    P(_titre("Graphes"))
    couleurs = plt.cm.viridis(np.linspace(0.05, 0.85, 6))

    def f_per(xv):
        t = af + np.mod(np.asarray(xv, dtype=float) - af, Tf)
        return _evalue_morceaux(pieces, x, t)

    def somme(xv, Nn):
        xv = np.asarray(xv, dtype=float)
        ks = np.arange(1, Nn + 1)[:, None]
        ang = ks * wf * xv[None, :]
        return A[0] / 2 + (A[1:Nn + 1, None] * np.cos(ang) + B[1:Nn + 1, None] * np.sin(ang)).sum(axis=0)

    zones_disc = [float(sp.N(d)) for d in disc]

    def coupe(xv, yv):
        """Insère des NaN autour des discontinuités pour ne pas relier les sauts."""
        yv = np.array(yv, dtype=float)
        if not zones_disc:
            return yv
        dx = (xv[-1] - xv[0]) / (len(xv) - 1)
        for z0 in zones_disc:
            k0 = int(np.floor((xv[0] - z0) / Tf)) - 1
            k1 = int(np.ceil((xv[-1] - z0) / Tf)) + 1
            for k in range(k0, k1 + 1):
                z = z0 + k * Tf
                yv[np.abs(xv - z) < 1.5 * dx] = np.nan
        return yv

    # --- 1) f et sommes partielles
    xs = np.linspace(af - Tf, bf + Tf, 4001)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(xs, coupe(xs, f_per(xs)), color="black", linewidth=2.2, label=r"$\tilde f$")
    liste = sorted(set([1, 3, max(1, N), min(K, 2 * max(1, N))]))
    for i, Nn in enumerate(liste):
        ax.plot(xs, somme(xs, Nn), color=couleurs[i], linewidth=1.4, label=rf"$S_{{{Nn}}}$")
    for k in range(-1, 3):
        ax.axvline(af + k * Tf, color="gray", linewidth=0.7, linestyle=":")
    _style_axes(ax, "Fonction périodisée et sommes partielles de Fourier", "x", "y")
    ax.legend(ncol=5, fontsize=9)
    st.pyplot(fig)

    # --- 2) spectre
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    Kp = min(K, max(12, N))
    ks = np.arange(1, Kp + 1)
    Ap = np.where(np.abs(A[1:Kp + 1]) < 1e-12, 0.0, A[1:Kp + 1])
    Bp = np.where(np.abs(B[1:Kp + 1]) < 1e-12, 0.0, B[1:Kp + 1])
    axs[0].bar(ks, Ap, color="#1f77b4")
    if not np.any(Ap):
        axs[0].set_ylim(-1, 1)
    _style_axes(axs[0], r"Coefficients $a_n$", "n")
    axs[1].bar(ks, Bp, color="#d62728")
    if not np.any(Bp):
        axs[1].set_ylim(-1, 1)
    _style_axes(axs[1], r"Coefficients $b_n$", "n")
    amp = np.sqrt(A[1:] ** 2 + B[1:] ** 2)
    kk = np.arange(1, K + 1)
    pos = amp > 1e-14
    axs[2].loglog(kk[pos], amp[pos], ".", color="#2ca02c", markersize=4, label=r"$\sqrt{a_n^2+b_n^2}=2|c_n|$")
    if p_ord is not None and p_ord >= 0:
        ref = amp[pos].max() * (kk[pos][0] / kk) ** p_ord if pos.any() else None
        if ref is not None:
            axs[2].loglog(kk, ref, "--", color="gray", label=rf"pente $-{p_ord}$")
    _style_axes(axs[2], "Spectre d'amplitude (échelle log-log)", "n")
    axs[2].legend(fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)

    # --- 3) erreurs en fonction de N
    Ns = np.unique(np.concatenate([np.arange(1, 31), np.array([40, 50, 70, 100, 150, 200, 300, 400])]))
    Ns = Ns[Ns <= K]
    cum = A[0] ** 2 / 4 + 0.5 * np.cumsum(A[1:] ** 2 + B[1:] ** 2)
    cum = np.concatenate([[A[0] ** 2 / 4], cum])
    err_l2 = np.sqrt(np.maximum(Ef_num - cum, 0.0))
    xg = np.linspace(af, bf, 3001)[:-1]
    fg = f_per(xg)
    if zones_disc:
        loin = np.ones_like(xg, dtype=bool)
        for z0 in zones_disc:
            dist = np.abs(((xg - z0 + Tf / 2) % Tf) - Tf / 2)
            loin &= dist > 0.05 * Tf
    else:
        loin = np.ones_like(xg, dtype=bool)
    sup_all, sup_loin = [], []
    for Nn in Ns:
        e = np.abs(fg - somme(xg, int(Nn)))
        sup_all.append(np.nanmax(e))
        sup_loin.append(np.nanmax(e[loin]) if loin.any() else np.nan)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    axs[0].loglog(Ns, np.maximum(err_l2[Ns], 1e-16), "o-", color="#9467bd", markersize=4, label=r"$\|f-S_N\|_2$ (Parseval)")
    axs[0].loglog(Ns, np.maximum(sup_all, 1e-16), "s-", color="#d62728", markersize=4, label=r"$\|f-S_N\|_\infty$ (sur toute la période)")
    if zones_disc:
        axs[0].loglog(Ns, np.maximum(sup_loin, 1e-16), "^-", color="#2ca02c", markersize=4, label=r"$\|f-S_N\|_\infty$ loin des sauts")
    _style_axes(axs[0], "Erreur en norme quadratique et en norme uniforme", "N")
    axs[0].legend(fontsize=8)
    frac = cum / Ef_num if Ef_num > 0 else cum
    axs[1].semilogx(np.arange(0, K + 1)[1:], np.clip(frac[1:], 0, 1.0001), color="#1f77b4")
    axs[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    _style_axes(axs[1], r"Part de l'énergie $\frac{1}{T}\int f^2$ capturée par $S_N$ (Parseval)", "N")
    axs[1].set_ylim(min(0.0, np.nanmin(frac[1:]) - 0.02), 1.05)
    fig.tight_layout()
    st.pyplot(fig)

    # --- 4) carte d'erreur |f - S_N(x)| (x en abscisse, N en ordonnée)
    Nmax = min(K, 60)
    xm = np.linspace(af, bf, 500)
    fm = f_per(xm)
    carte = np.empty((Nmax, len(xm)))
    ang_all = wf * xm[None, :]
    c_part = A[0] / 2 * np.ones_like(xm)
    for Nn in range(1, Nmax + 1):
        c_part = c_part + A[Nn] * np.cos(Nn * ang_all[0]) + B[Nn] * np.sin(Nn * ang_all[0])
        carte[Nn - 1] = np.abs(fm - c_part)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    im = ax.imshow(np.log10(carte + 1e-12), aspect="auto", origin="lower", extent=[af, bf, 1, Nmax], cmap="magma")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}|f(x)-S_N(x)|$")
    _style_axes(ax, "Carte de l'erreur : où et à quelle vitesse les sommes partielles convergent", "x", "N")
    st.pyplot(fig)

    # --- 5) phénomène de Gibbs (si saut) ou erreur ponctuelle
    if sauts or (not rac_ok):
        x0 = zones_disc[0] if zones_disc else af
        g, d = _limites_en(pieces, x, a, b, T, sp.nsimplify(x0, rational=True) if not zones_disc else disc[0])
        gf, df_ = (float(sp.N(g)), float(sp.N(d))) if g is not None and d is not None else (np.nan, np.nan)
        delta = df_ - gf
        h = 0.12 * Tf
        xz = np.linspace(x0 - h, x0 + h, 3001)
        fig, ax = plt.subplots(figsize=(11, 4.6))
        ax.plot(xz, coupe(xz, f_per(xz)), color="black", linewidth=2.2, label=r"$\tilde f$")
        for i, Nn in enumerate(sorted(set([max(1, N), min(K, 4 * max(1, N)), min(K, 16 * max(1, N))]))):
            ax.plot(xz, somme(xz, Nn), color=couleurs[i + 1], linewidth=1.3, label=rf"$S_{{{Nn}}}$")
        if np.isfinite(delta):
            haut = max(gf, df_) + 0.0895 * abs(delta)
            bas = min(gf, df_) - 0.0895 * abs(delta)
            ax.axhline(haut, color="#d62728", linestyle="--", linewidth=0.9, label="limite de Gibbs : saut $\\pm 8{,}95\\%$")
            ax.axhline(bas, color="#d62728", linestyle="--", linewidth=0.9)
            ax.axhline((gf + df_) / 2, color="gray", linestyle=":", linewidth=0.9, label=r"$(\tilde f(x_0^+)+\tilde f(x_0^-))/2$")
        _style_axes(ax, r"Phénomène de Gibbs au voisinage d'un saut ($x_0\approx %.3g$)" % x0, "x", "y")
        ax.legend(ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        st.pyplot(fig)
    else:
        xe = np.linspace(af, bf, 2001)
        fe = f_per(xe)
        fig, ax = plt.subplots(figsize=(11, 4.2))
        for i, Nn in enumerate(sorted(set([1, 3, max(1, N), min(K, 4 * max(1, N))]))):
            ax.semilogy(xe, np.abs(fe - somme(xe, Nn)) + 1e-16, color=couleurs[i], linewidth=1.2, label=rf"$|f-S_{{{Nn}}}|$")
        _style_axes(ax, "Erreur ponctuelle sur une période", "x", "erreur (échelle log)")
        ax.legend(ncol=4, fontsize=9)
        st.pyplot(fig)

