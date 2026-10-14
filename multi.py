# =============================================================
# multi.py : intégrales multiples (Fubini, Fubini–Tonelli, changement de variables)
#   domaine donné par des inégalités, dimension quelconque
# =============================================================
import re
import time
import itertools
import random
import numpy as np
import sympy as sp
from sympy import S

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lebesgue as _L

st = None  # renseigné par app.py : multi.st = st


class _NonSupporte(Exception):
    pass


def _push():
    _L.st = st


_ALERTE = []
_MAUVAIS = re.compile(r"NaN|\bnan\b|zoo|\\tilde\{\\infty\}|\bNone\b")


def _P(s):
    _push()
    if _MAUVAIS.search(s):
        _ALERTE.append(s)   # jamais d'affichage d'une valeur indéfinie
        return
    _L._P(s)


def _T(s):
    return _L._T(s)


def _titre(s):
    _push()
    _L._titre(s)


def _rouge(s):
    _push()
    _L._rouge(s)


def _delai(fn, sec=12):
    return _L._avec_delai(fn, sec)


# -------------------------------------------------------------
# Analyse syntaxique : fonction, variables, domaine
# -------------------------------------------------------------
_RESERVES = {"oo", "inf", "in", "et", "and", "Abs", "abs", "Min", "Max", "min", "max", "sign", "Piecewise"}


_AFF = []


def _decouper_top(s, seps=",;"):
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


def _normaliser_texte(s):
    s = s or ""
    s = s.replace("≤", "<=").replace("≥", ">=").replace("−", "-").replace("∞", "oo").replace("×", "*")
    s = s.replace("=<", "<=").replace("=>", ">=").replace("&&", ",").replace("∧", ",")
    s = re.sub(r"\bet\b", ",", s)
    s = re.sub(r"\band\b", ",", s)
    s = re.sub(r"\binf\b", "oo", s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\bD\s*=\s*", "", s)
    s = s.replace("^", "**")
    return s


def _intervalle_vers_chaine(m):
    var, g, a, b, d = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    op_g = "<=" if g == "[" else "<"
    op_d = "<=" if d == "]" and True else "<"
    if g == "]":
        op_g = "<"
    if d == "[":
        op_d = "<"
    return "%s %s %s %s %s" % (a, op_g, var, op_d, b)


def _prepare_domaine(s):
    s = _normaliser_texte(s)
    # ℝ entier : « x in R », « x in R+ », « R^2 », « R**n », « R » seul
    s = s.replace("ℝ", "R").replace("²", "**2").replace("³", "**3")
    def _multi_var(m):
        return ", ".join("-oo<%s<oo" % v.strip() for v in m.group(1).split(","))
    s = re.sub(r"\b([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)+)\s*(?:in|∈)\s*R(?![\w(\[+])(?:\s*(?:\*\*|\^)\s*\d+)?", _multi_var, s)
    s = re.sub(r"\b([A-Za-z_]\w*)\s*(?:in|∈)\s*R\s*(?:\+|_\+|_\{\+\}|plus)(?![\w*])", r"\1>=0", s)
    s = re.sub(r"\b([A-Za-z_]\w*)\s*(?:in|∈)\s*R(?![\w(\[])", r"-oo<\1<oo", s)
    s = re.sub(r"(?<![\w.])R\s*(?:\*\*|\^)?\s*(?:\d+|n|k|d)?(?![\w(\[])", "", s)
    s = re.sub(r"^[\s,;]+|[\s,;]+$", "", s)
    s = re.sub(r"[,;]\s*[,;]+", ",", s)
    pat = re.compile(r"([A-Za-z_]\w*)\s*(?:in|∈)\s*([\[\]\(])\s*([^,;\[\]\(\)]+?)\s*,\s*([^,;\[\]\(\)]+?)\s*([\[\]\)])")
    s = pat.sub(_intervalle_vers_chaine, s)
    # produit cartésien d'intervalles : [0,1]x[0,2] (avec variables positionnelles) non géré ici
    s = re.sub(r"\|([^|]+)\|", r"Abs(\1)", s)
    return s


_REL = re.compile(r"(<=|>=|<|>)")


def _parse_contraintes(txt, loc):
    """Retourne une liste de contraintes (g, strict) signifiant g <= 0 (ou g < 0)."""
    cons = []
    _AFF.clear()
    for piece in _decouper_top(_prepare_domaine(txt)):
        if re.search(r"(?<![<>=!])=(?!=)", piece) and not _REL.search(piece):
            raise _NonSupporte("Une égalité (« = ») définit une courbe ou une surface, de mesure nulle : écrire des inégalités.")
        toks = _REL.split(piece)
        if len(toks) < 3:
            raise _NonSupporte("Contrainte non reconnue : « %s »." % piece)
        exprs = [sp.sympify(e.strip(), locals=loc) for e in toks[0::2]]
        ops = toks[1::2]
        _lat_op = {"<=": r" \le ", ">=": r" \ge ", "<": " < ", ">": " > "}
        _AFF.append("".join(sp.latex(exprs[i]) + _lat_op[ops[i]] for i in range(len(ops))) + sp.latex(exprs[-1]))
        for i, op in enumerate(ops):
            a, b = exprs[i], exprs[i + 1]
            if a in (sp.oo, -sp.oo) or b in (sp.oo, -sp.oo):
                # borne infinie : contrainte vide ou impossible
                if (a == -sp.oo and op in ("<", "<=")) or (a == sp.oo and op in (">", ">=")) or (b == sp.oo and op in ("<", "<=")) or (b == -sp.oo and op in (">", ">=")):
                    continue
                raise _NonSupporte("Contrainte impossible : « %s »." % piece)
            strict = op in ("<", ">")
            g = (a - b) if op in ("<", "<=") else (b - a)
            cons.append((g, strict))
    return cons


def _noms_utilisables(chaine, loc):
    ids = set(re.findall(r"[A-Za-z_]\w*", chaine))
    return {i for i in ids if i not in loc and i not in _RESERVES}


def _variables_positives(cons, V):
    """Variables v pour lesquelles une contrainte affine impose v >= 0 (ou v > 0)."""
    pos = set()
    for g, _s in cons:
        for v in V:
            try:
                if g.free_symbols == {v}:
                    P = sp.Poly(g, v)
                    if P.degree() == 1:
                        a_, b_ = P.all_coeffs()
                        if a_.is_number and b_.is_number and a_ < 0 and b_ >= 0:
                            pos.add(v)
            except Exception:
                pass
    return pos


def _effacer_denominateurs(cons, V):
    """Multiplie une contrainte rationnelle par son dénominateur lorsque celui-ci est positif (variables > 0)."""
    if not any(sp.denom(sp.together(g)).free_symbols for g, _ in cons):
        return cons
    pos = _variables_positives(cons, V)
    out = []
    for g, s in cons:
        n, d = sp.fraction(sp.together(g))
        if d.free_symbols:
            monome = all(isinstance(b_, sp.Symbol) for b_, _e in d.as_powers_dict().items() if b_.free_symbols)
            if monome and all(sy in pos for sy in d.free_symbols):
                g = sp.expand(n)
            else:
                raise _NonSupporte("Une contrainte contient un dénominateur (%s) dont le signe n'est pas connu : ajouter par exemple « %s>0 » au domaine." % (sp.latex(d), sp.latex(sorted(d.free_symbols, key=str)[0])))
        out.append((g, s))
    return out


def _analyser(fstr, dstr, vstr, loc_app):
    """Retourne (f, cons, V) avec V = liste ordonnée de symboles réels."""
    _AFF.clear()
    loc0 = dict(loc_app or {})
    loc0.setdefault("oo", sp.oo)
    loc0.setdefault("Abs", sp.Abs)
    loc0.setdefault("abs", sp.Abs)
    loc0.setdefault("Min", sp.Min)
    loc0.setdefault("Max", sp.Max)
    loc0.setdefault("min", sp.Min)
    loc0.setdefault("max", sp.Max)
    loc0.pop("x", None)
    loc0.pop("n", None)
    fstr = _normaliser_texte(fstr)
    dtxt = _prepare_domaine(dstr)
    if vstr and vstr.strip():
        noms = [t for t in re.split(r"[\s,;]+", vstr.strip()) if t]
    else:
        noms = sorted(_noms_utilisables(fstr + " " + dtxt, loc0), key=lambda t: (t not in "xyz", t))
    if not noms:
        raise _NonSupporte("Aucune variable détectée : la fonction doit dépendre d'au moins une variable (x, y, z, ...).")
    V = [sp.Symbol(t, real=True) for t in noms]
    loc = dict(loc0)
    for v in V:
        loc[v.name] = v
    inconnus = _noms_utilisables(fstr + " " + dtxt, loc) - set(noms)
    if inconnus:
        raise _NonSupporte("Symbole non défini : %s (variables attendues : %s)." % (", ".join(sorted(inconnus)), ", ".join(noms)))
    f = sp.sympify(fstr, locals=loc)
    cons = _parse_contraintes(dtxt, loc) if dtxt.strip() else []
    cons = _effacer_denominateurs(cons, V)
    return f, cons, V


# -------------------------------------------------------------
# Découpage selon les valeurs absolues / Min / Max / signe
# -------------------------------------------------------------
def _atomes_scindables(e, V):
    res = []
    for a in e.atoms(sp.Abs, sp.Max, sp.Min, sp.sign):
        if any(a.has(v) for v in V):
            res.append(a)
    # intérieurs d'abord
    res.sort(key=lambda a: len(a.atoms(sp.Abs, sp.Max, sp.Min, sp.sign)))
    return res


def _cas_atome(a):
    """[(substitut, [contraintes]), ...] disjoints."""
    if isinstance(a, sp.Abs):
        h = a.args[0]
        return [(h, [(-h, False)]), (-h, [(h, True)])]
    if isinstance(a, (sp.Max, sp.Min)):
        x0 = a.args[0]
        reste = type(a)(*a.args[1:]) if len(a.args) > 2 else a.args[1]
        if isinstance(a, sp.Max):
            return [(x0, [(reste - x0, False)]), (reste, [(x0 - reste, True)])]
        return [(x0, [(x0 - reste, False)]), (reste, [(reste - x0, True)])]
    if isinstance(a, sp.sign):
        h = a.args[0]
        return [(sp.Integer(1), [(-h, True)]), (sp.Integer(-1), [(h, True)])]
    return None


def _abs_convexe(cons):
    """Abs(h) + reste <= 0  ->  deux contraintes convexes (sans scinder le domaine)."""
    res = []
    for g, strict in cons:
        done = False
        if g.is_Add:
            absl = [t for t in g.args if isinstance(t, sp.Abs)]
            if len(absl) == 1 and not (g - absl[0]).has(sp.Abs):
                h = absl[0].args[0]
                reste = g - absl[0]
                if not h.has(sp.Abs):
                    res.append((h + reste, strict))
                    res.append((-h + reste, strict))
                    done = True
        elif isinstance(g, sp.Abs):
            h = g.args[0]
            res.append((h, strict))
            res.append((-h, strict))
            done = True
        if not done:
            res.append((g, strict))
    return res


def _scinder(f, cons, V, prof=0):
    """Liste de morceaux disjoints dict(f, cons) sans Abs/Min/Max/sign dans f ni dans les contraintes."""
    cons = _abs_convexe(cons)
    tous = [f] + [g for g, _ in cons]
    atome = None
    for e in tous:
        at = _atomes_scindables(e, V)
        if at:
            atome = at[0]
            break
    if atome is None:
        return [dict(f=f, cons=cons)]
    if prof >= 6:
        raise _NonSupporte("Trop de valeurs absolues / min / max à découper.")
    morceaux = []
    for subst, extra in _cas_atome(atome):
        f2 = f.subs(atome, subst)
        c2 = [(g.subs(atome, subst), s) for g, s in cons] + extra
        morceaux += _scinder(f2, c2, V, prof + 1)
    return morceaux


# -------------------------------------------------------------
# Échantillonnage
# -------------------------------------------------------------
_RNG = np.random.RandomState(12345)


def _lamb(exprs, V):
    try:
        return sp.lambdify(V, exprs, "numpy")
    except Exception:
        return None


def _masque(cons, V, P):
    """P : tableau (N, k) ; booléens des points vérifiant toutes les contraintes."""
    ok = np.ones(P.shape[0], dtype=bool)
    cols = [P[:, i] for i in range(P.shape[1])]
    for g, strict in cons:
        if g.is_number:
            val = float(g)
            if (val < 0) if strict else (val <= 0):
                continue
            return np.zeros(P.shape[0], dtype=bool)
        fn = _lamb(g, V)
        if fn is None:
            return np.zeros(P.shape[0], dtype=bool)
        with np.errstate(all="ignore"):
            try:
                v = np.asarray(fn(*cols), dtype=float) * np.ones(P.shape[0])
            except Exception:
                return np.zeros(P.shape[0], dtype=bool)
        ok &= (v < 0) if strict else (v <= 0)
    return ok


def _points_dans(cons, V, n=400, essais=40000, rayons=(1.5, 4.0, 12.0, 40.0), rng=None):
    """Points vérifiant les contraintes, tirés à plusieurs échelles (pour ne pas manquer de régime)."""
    rng = rng or _RNG
    k = len(V)
    pts = []
    par = max(n // len(rayons), 60)
    for R in rayons:
        P = rng.uniform(-R, R, size=(essais, k))
        m = _masque(cons, V, P)
        if m.any():
            pts.append(P[m][: par])
    if not pts:
        return np.zeros((0, k))
    return np.vstack(pts)


# -------------------------------------------------------------
# Bornes d'une variable à partir de contraintes
# -------------------------------------------------------------
def _ensemble_1d(cons1, v):
    """Intersection d'inégalités ne portant que sur v : Union d'intervalles."""
    ens = sp.Reals
    for g, strict in cons1:
        rel = (g < 0) if strict else (g <= 0)
        try:
            sol = sp.solve_univariate_inequality(rel, v, relational=False)
        except Exception:
            raise _NonSupporte("Inégalité non résolue en %s : %s" % (v, sp.latex(rel)))
        ens = sp.Intersection(ens, sol)
    return ens


def _alternatives_contrainte(g, strict, v):
    """Pour une contrainte g(v, autres) <= 0, liste d'alternatives dict(lo, hi, extra)."""
    try:
        P = sp.Poly(sp.expand(g), v)
    except Exception:
        raise _NonSupporte("Contrainte non polynomiale en %s : %s" % (v, sp.latex(g)))
    d = P.degree()
    co = P.all_coeffs()
    if d == 1:
        a, b = co
        if a.is_number:
            if a > 0:
                return [dict(lo=[], hi=[sp.simplify(-b / a)], extra=[])]
            return [dict(lo=[sp.simplify(-b / a)], hi=[], extra=[])]
        raise _NonSupporte("Coefficient de %s non constant dans %s." % (v, sp.latex(g)))
    if d == 2:
        a, b, c = co
        if not a.is_number:
            raise _NonSupporte("Coefficient de %s^2 non constant dans %s." % (v, sp.latex(g)))
        D = sp.expand(b ** 2 - 4 * a * c)
        sq = sp.sqrt(sp.factor(D)) if not D.is_number else sp.sqrt(D)
        r1 = sp.simplify((-b - sq) / (2 * a))
        r2 = sp.simplify((-b + sq) / (2 * a))
        if a > 0:
            return [dict(lo=[r1], hi=[r2], extra=[(-D, False)])]
        # a < 0 : g <= 0 hors des racines
        rp, rg = (r2, r1)  # a<0 : r2 < r1
        return [dict(lo=[], hi=[rp], extra=[(-D, False)]),
                dict(lo=[rg], hi=[], extra=[(-D, False)]),
                dict(lo=[], hi=[], extra=[(D, True)])]
    raise _NonSupporte("Contrainte de degré %d en %s non prise en charge : %s" % (d, v, sp.latex(g)))


def _dedup(lst):
    res = []
    for e in lst:
        if not any(sp.simplify(e - r) == 0 for r in res):
            res.append(e)
    return res


def _num(e):
    try:
        return float(e)
    except Exception:
        return None


def _reduire_const(lst, maxi):
    nums = [e for e in lst if e.is_number and e.is_finite]
    autres = [e for e in lst if not (e.is_number and e.is_finite)]
    if nums:
        m = max(nums, key=lambda t: float(t)) if maxi else min(nums, key=lambda t: float(t))
        autres.append(m)
    return autres


# -------------------------------------------------------------
# Construction des cellules (domaines « simples » emboîtés)
# -------------------------------------------------------------
def _eval_bornes(exprs, rest, P):
    """Valeurs numériques (len(exprs), N) de bornes dépendant des variables rest aux points P (N, len(rest))."""
    N = P.shape[0] if len(rest) else 1
    out = np.empty((len(exprs), N))
    for i, e in enumerate(exprs):
        if len(rest) == 0:
            try:
                out[i, :] = float(e)
            except Exception:
                out[i, :] = np.nan
            continue
        fn = _lamb(e, rest)
        with np.errstate(all="ignore"):
            try:
                v = np.asarray(fn(*[P[:, j] for j in range(len(rest))]), dtype=float)
            except Exception:
                v = np.full(N, np.nan)
        out[i, :] = v * np.ones(N)
    return out


def _construire(C, outers, t0, budget):
    if time.time() - t0 > budget:
        raise _NonSupporte("Délai dépassé pour la description du domaine.")
    if not outers:
        for g, s in C:
            if not g.is_number:
                raise _NonSupporte("Contrainte non réduite : %s" % sp.latex(g))
            val = float(g)
            if not ((val < 0) if s else (val <= 0)):
                return []
        return [[]]
    v = outers[-1]
    rest = outers[:-1]
    Cv = [(g, s) for g, s in C if g.has(v)]
    Co = [(g, s) for g, s in C if not g.has(v)]
    single = [(g, s) for g, s in Cv if g.free_symbols <= {v}]
    multi = [(g, s) for g, s in Cv if not (g.free_symbols <= {v})]
    ens = _ensemble_1d(single, v) if single else sp.Reals
    if ens is sp.S.EmptySet or ens == sp.S.EmptySet:
        return []
    ivs = list(ens.args) if isinstance(ens, sp.Union) else [ens]
    ivs = [i for i in ivs if isinstance(i, sp.Interval)]
    set_alts = []
    for iv in ivs:
        set_alts.append(dict(lo=[iv.start] if iv.start != -sp.oo else [], hi=[iv.end] if iv.end != sp.oo else [], extra=[]))
    if not set_alts:
        return []
    listes = [set_alts] + [_alternatives_contrainte(g, s, v) for g, s in multi]
    out = []
    n_comb = 1
    for l in listes:
        n_comb *= len(l)
    if n_comb > 64:
        raise _NonSupporte("Description du domaine trop complexe.")
    for combi in itertools.product(*listes):
        lows = _dedup(_reduire_const(sum([a["lo"] for a in combi], []), True))
        ups = _dedup(_reduire_const(sum([a["hi"] for a in combi], []), False))
        extra = sum([a["extra"] for a in combi], [])
        Co2 = Co + extra
        if rest:
            P = _points_dans(Co2, rest, n=300)
            if len(P) == 0:
                continue
        else:
            P = np.zeros((1, 0))
            ok = True
            for g, s in Co2:
                if g.is_number:
                    val = float(g)
                    if not ((val < 0) if s else (val <= 0)):
                        ok = False
                else:
                    ok = False
            if not ok:
                continue
        lv = _eval_bornes(lows, rest, P) if lows else None
        uv = _eval_bornes(ups, rest, P) if ups else None
        win_l = [None]
        win_u = [None]
        if lows:
            lv2 = np.where(np.isnan(lv), -np.inf, lv)
            win_l = sorted(set(np.argmax(lv2, axis=0).tolist()))
        if ups:
            uv2 = np.where(np.isnan(uv), np.inf, uv)
            win_u = sorted(set(np.argmin(uv2, axis=0).tolist()))
        for a in win_l:
            for b in win_u:
                cond = []
                msk = np.ones(P.shape[0], dtype=bool)
                if lows and len(win_l) > 1:
                    cond += [(lows[i] - lows[a], False) for i in win_l if i != a]
                if ups and len(win_u) > 1:
                    cond += [(ups[b] - ups[j], False) for j in win_u if j != b]
                if lows and a is not None:
                    msk &= (np.argmax(np.where(np.isnan(lv), -np.inf, lv), axis=0) == a)
                if ups and b is not None:
                    msk &= (np.argmin(np.where(np.isnan(uv), np.inf, uv), axis=0) == b)
                La = lows[a] if a is not None else -sp.oo
                Ub = ups[b] if b is not None else sp.oo
                if a is not None and b is not None:
                    dif = lv[a] - uv[b]
                    sel = dif[msk] if msk.any() else dif
                    if np.all(sel <= 1e-12):
                        pass
                    elif np.all(sel > 0):
                        continue
                    else:
                        cond.append((La - Ub, False))
                newC = Co2 + cond
                for cell in _construire(newC, rest, t0, budget):
                    out.append(cell + [(v, La, Ub)])
    return out


# -------------------------------------------------------------
# Vérification : les cellules recouvrent exactement le domaine
# -------------------------------------------------------------
def _cellule_contient(cell, V, P):
    ok = np.ones(P.shape[0], dtype=bool)
    cols = [P[:, i] for i in range(P.shape[1])]
    for v, lo, hi in cell:
        j = V.index(v)
        for b, sg in ((lo, -1), (hi, 1)):
            if b in (sp.oo, -sp.oo):
                continue
            fn = _lamb(b, V)
            with np.errstate(all="ignore"):
                bv = np.asarray(fn(*cols), dtype=float) * np.ones(P.shape[0])
            ok &= (cols[j] >= bv) if sg < 0 else (cols[j] <= bv)
    return ok


def _partition_ok(morceaux_cells, pieces, V, nb=2500):
    """Chaque point de l'espace appartient à au plus une cellule, et exactement une si et seulement s'il est dans D."""
    rng = np.random.RandomState(7)
    bad = 0
    tot = 0
    k = len(V)
    for R in (1.5, 4.0, 12.0):
        P = rng.uniform(-R, R, size=(nb, k))
        vrai = np.zeros(nb, dtype=bool)
        for pc in pieces:
            vrai |= _masque(pc["cons"], V, P)
        cnt = np.zeros(nb, dtype=int)
        for cell in morceaux_cells:
            cnt += _cellule_contient(cell, V, P).astype(int)
        bad += int(np.sum(cnt != vrai.astype(int)))
        tot += nb
    return bad <= max(2, int(0.003 * tot))


def _cellules_ordre(pieces, ordre, V, budget=8.0):
    """Liste de (f, cellule) couvrant D pour l'ordre (extérieur -> intérieur)."""
    t0 = time.time()
    res = []
    for pc in pieces:
        for cell in _construire(list(pc["cons"]), list(ordre), t0, budget):
            res.append((pc["f"], [(v, sp.simplify(lo) if lo not in (sp.oo, -sp.oo) else lo, sp.simplify(hi) if hi not in (sp.oo, -sp.oo) else hi) for v, lo, hi in cell]))
    return res


def _score_cellules(cells):
    ops = 0
    for f, cell in cells:
        for v, lo, hi in cell:
            for b in (lo, hi):
                if b not in (sp.oo, -sp.oo):
                    ops += sp.count_ops(b)
    return (len(cells), ops)


def _ordres_possibles(pieces, V, budget=18.0, maximum=6):
    t0 = time.time()
    perms = list(itertools.permutations(V))
    if len(V) > 4:
        perms = [tuple(V), tuple(reversed(V))] + [tuple(V[i:] + V[:i]) for i in range(1, len(V))]
    res = []
    echec = None
    for p in perms:
        if res and time.time() - t0 > budget:
            break
        if not res and time.time() - t0 > budget * 2:
            break
        try:
            cells = _cellules_ordre(pieces, p, V)
        except _NonSupporte as e:
            echec = echec or str(e)
            continue
        except Exception as e:
            echec = echec or str(e)
            continue
        if not cells:
            continue
        if not _partition_ok([c for _, c in cells], pieces, V):
            echec = echec or "Le découpage en cellules n'a pas pu être validé."
            continue
        res.append((_score_cellules(cells), p, cells))
    res.sort(key=lambda t: (t[0], 0 if tuple(t[1]) == tuple(V) else 1))
    return res[:maximum], echec


# -------------------------------------------------------------
# Échantillons dans une cellule
# -------------------------------------------------------------
def _borne_num(b, ctx, defaut):
    if b == sp.oo or b == -sp.oo:
        return defaut
    try:
        return float(b.subs(ctx))
    except Exception:
        return None


def _echant_cellule(cell, m=6, seed=3, jusqua=None):
    """Points (dict variable -> float) de l'intérieur de la cellule ; jusqua : nombre de niveaux (extérieurs) à tirer."""
    rng = np.random.RandomState(seed)
    niveaux = cell if jusqua is None else cell[:jusqua]
    pts = []
    for _ in range(m * 4):
        ctx = {}
        ok = True
        for v, lo, hi in niveaux:
            a = _borne_num(lo, ctx, None)
            b = _borne_num(hi, ctx, None)
            if lo == -sp.oo and hi == sp.oo:
                a, b = -3.0, 3.0
            elif lo == -sp.oo:
                b = _borne_num(hi, ctx, None)
                a = None if b is None else b - 3.0
            elif hi == sp.oo:
                a = _borne_num(lo, ctx, None)
                b = None if a is None else a + 3.0
            if a is None or b is None or not (b > a):
                ok = False
                break
            ctx[v] = float(rng.uniform(a, b))
        if ok:
            pts.append(ctx)
        if len(pts) >= m:
            break
    return pts


# -------------------------------------------------------------
# Intégrale en une variable avec paramètres
# -------------------------------------------------------------
_SPEC = (sp.Integral, sp.meijerg, sp.hyper, sp.lerchphi, sp.polylog, sp.uppergamma, sp.lowergamma, sp.Ei, sp.li, sp.Piecewise)
_SPECF = (sp.erf, sp.erfc, sp.erfi, sp.Si, sp.Ci, sp.Shi, sp.Chi, sp.Ei, sp.li, sp.fresnels, sp.fresnelc, sp.expint, sp.lowergamma, sp.uppergamma, sp.polylog, sp.lerchphi, sp.hyper, sp.meijerg)


def _signes_params(outers, samples):
    sg = {}
    for o in outers:
        vals = [s[o] for s in samples if o in s]
        if vals and all(t > 0 for t in vals):
            sg[o] = "+"
        elif vals and all(t < 0 for t in vals):
            sg[o] = "-"
    return sg


def _val_num_1d(g, v, lo, hi, s):
    try:
        a = lo.subs(s) if lo not in (sp.oo, -sp.oo) else lo
        b = hi.subs(s) if hi not in (sp.oo, -sp.oo) else hi
        gg = g.subs(s)
        fm = _L._fnum(gg, v)
        if fm is None:
            return None
        return _L._quad(fm, a, b)
    except Exception:
        return None


def _choisir_branche(val, samples):
    if not isinstance(val, sp.Piecewise):
        return val
    for e_, c_ in val.args:
        try:
            if all(bool(c_.subs(s)) for s in samples):
                return e_
        except Exception:
            pass
    return None



# -------------------------------------------------------------
# Intégrales de type Gauss : C * P(v) * exp(-a v^2 + b v), bornes infinies
# -------------------------------------------------------------
def _forme_gauss(g, v):
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


def _gauss_valeur(g, v, lo, hi):
    """Valeur exacte (sans fonction d'erreur) d'une intégrale de type Gauss, ou None."""
    fg = _forme_gauss(g, v)
    if fg is None:
        return None
    C, P, a, b = fg
    try:
        if b == 0 and all(j % 2 == 1 for (j,), _c in sp.Poly(P, v).terms()):
            return None   # primitive élémentaire (changement u = v^2) : pas besoin du lemme de Gauss
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


def _int1(g, v, lo, hi, outers, samples):
    """Intégrale de g en v entre lo et hi (paramètres : outers). Renvoie dict(F, val) ou None."""
    if lo == hi or g == 0:
        return dict(F=None, val=sp.Integer(0))
    sg = _signes_params(outers, samples)
    rep, inv = {}, {}
    for o in outers:
        if o in sg:
            d = sp.Symbol(o.name, positive=(sg[o] == "+"), negative=(sg[o] == "-"))
            rep[o] = d
            inv[d] = o
    g2 = g.subs(rep)
    lo2 = lo.subs(rep) if lo not in (sp.oo, -sp.oo) else lo
    hi2 = hi.subs(rep) if hi not in (sp.oo, -sp.oo) else hi
    gv = _gauss_valeur(g2, v, lo2, hi2)
    F = None if gv is not None else _delai(lambda: sp.integrate(g2, v), 5)
    val = gv
    if isinstance(F, sp.Piecewise):
        F = _choisir_branche(F, [dict((rep.get(k, k), x_) for k, x_ in s.items()) for s in samples])
    if F is not None and F.has(sp.polar_lift):
        return None
    if F is not None and F.has(*_SPECF):
        F = None   # primitive non élémentaire : interdite (on tente l'intégrale définie par d'autres moyens)
    if F is not None and not F.has(*_SPEC) or (F is not None and not F.has(sp.Integral, sp.Piecewise, sp.meijerg, sp.hyper, sp.lerchphi, sp.polylog, sp.Ei, sp.li)):
        _ab = _delai(lambda: (_L._valeur_F(F, v, lo2, "+"), _L._valeur_F(F, v, hi2, "-")), 6)
        if _ab is None:
            return None
        Fa, Fb = _ab
        if Fa[0] == "fin" and Fb[0] == "fin":
            val = sp.simplify(Fb[1] - Fa[1])
        elif Fa[0] in ("fin", "inf") and Fb[0] in ("fin", "inf") and (Fa[0] == "inf" or Fb[0] == "inf"):
            if Fb[0] == "inf" and Fa[0] == "fin":
                val = Fb[1]
            elif Fa[0] == "inf" and Fb[0] == "fin":
                val = -Fa[1]
            elif Fa[0] == "inf" and Fb[0] == "inf" and Fa[1] != Fb[1]:
                val = Fb[1] - Fa[1]
        if val is not None and val.has(sp.nan, sp.zoo):
            val = None
    F_elementaire = (F is not None)
    if val is None and F_elementaire:
        R = _delai(lambda: sp.integrate(g2, (v, lo2, hi2)), 5)
        if R is not None:
            R = _choisir_branche(R, [dict((rep.get(k, k), x_) for k, x_ in s.items()) for s in samples])
        if R is not None and not R.has(sp.Integral, sp.Piecewise, sp.nan, sp.zoo, *_SPECF):
            val = sp.simplify(R)
            F = None
    if val is None:
        return None
    if val.has(*_SPECF):
        return None
    val = val.subs(inv)
    if F is not None:
        F = F.subs(inv)
    # validation numérique
    if val not in (sp.oo, -sp.oo):
        bons = 0
        for s in samples[:3]:
            vn = _val_num_1d(g, v, lo, hi, s)
            try:
                ve = complex(sp.N(val.subs(s)))
            except Exception:
                return None
            if vn is None:
                continue
            if abs(ve.imag) > 1e-8 or abs(ve.real - vn) > 1e-5 * (1 + abs(vn)):
                return None
            bons += 1
    return dict(F=F, val=val)


def _calcul_cellule(f, cell, deadline=None):
    """Intégrale itérée de f sur une cellule (extérieur -> intérieur). Renvoie (valeur, étapes) ou (None, étapes)."""
    steps = []
    g = f
    for idx in range(len(cell) - 1, -1, -1):
        if deadline is not None and time.time() > deadline:
            return None, steps
        v, lo, hi = cell[idx]
        outers = [c[0] for c in cell[:idx]]
        samples = _echant_cellule(cell, m=4, jusqua=idx)
        if idx > 0 and not samples:
            return None, steps
        if idx == 0:
            samples = [dict()]
        r = _int1(g, v, lo, hi, outers, samples)
        if r is None:
            steps.append(dict(echec=True, v=v, lo=lo, hi=hi, g=g, outers=outers))
            return None, steps
        steps.append(dict(v=v, lo=lo, hi=hi, g=g, F=r["F"], val=r["val"], outers=outers))
        g = r["val"]
        if g in (sp.oo, -sp.oo):
            return g, steps
    return g, steps


# -------------------------------------------------------------
# Changements de variables : polaires, cylindriques, sphériques (centrés ou décalés, ellipses)
# -------------------------------------------------------------
def _coeffs_groupe(g, grp):
    """{'sq': [A_i], 'cross': bool, 'lin': [D_i], 'cst': F} si g est de degré <= 2 en grp, sinon None."""
    try:
        P = sp.Poly(sp.expand(g), *grp)
    except Exception:
        return None
    if P.total_degree() > 2:
        return None
    n = len(grp)
    d = P.as_dict()
    sq = [d.get(tuple(2 if j == i else 0 for j in range(n)), sp.Integer(0)) for i in range(n)]
    lin = [d.get(tuple(1 if j == i else 0 for j in range(n)), sp.Integer(0)) for i in range(n)]
    cst = d.get(tuple(0 for _ in range(n)), sp.Integer(0))
    cross = any(m for m, c in d.items() if sum(m) == 2 and max(m) == 1 and c != 0)
    return dict(sq=sq, lin=lin, cst=cst, cross=cross)


def _arc_demiplan(alpha, beta, signe=1):
    """Arc {theta : signe*(alpha cos t + beta sin t) >= 0} = [s, s+pi] ; renvoie s (symbolique)."""
    a, b = signe * alpha, signe * beta
    phi = sp.atan2(b, a)
    phi = sp.nsimplify(phi) if phi.is_number else phi
    return sp.simplify(phi - sp.pi / 2)


def _inter_arcs(starts):
    """Intersection d'arcs [s_i, s_i + pi] : (lo, hi) ou None (vide)."""
    if not starts:
        return (sp.Integer(0), 2 * sp.pi)
    s0 = starts[0]
    los, his = [s0], [s0 + sp.pi]
    for s in starts[1:]:
        d = float(s - s0)
        k = round(d / (2 * np.pi))
        s2 = s - 2 * sp.pi * k
        los.append(s2)
        his.append(s2 + sp.pi)
    lo = max(los, key=lambda t: float(t))
    hi = min(his, key=lambda t: float(t))
    if float(hi) - float(lo) <= 1e-12:
        return None
    return (sp.simplify(lo), sp.simplify(hi))


def _noms_libres(V, base):
    noms = {v.name for v in V}
    res = []
    for b in base:
        nm = b
        while nm in noms:
            nm += "_"
        noms.add(nm)
        res.append(sp.Symbol(nm, real=True))
    return res


def _essai_changement(f, pieces, V):
    """Tente polaires (2 variables) ou sphériques (3 variables). Renvoie une description ou None."""
    k = len(V)
    groupes = []
    for taille in (3, 2):
        if k >= taille:
            for grp in itertools.combinations(V, taille):
                groupes.append(tuple(grp))
    for grp in groupes:
        r = _essai_groupe(f, pieces, V, grp)
        if r is not None:
            return r
    return None


def _essai_groupe(f, pieces, V, grp):
    n = len(grp)
    ref = None
    for pc in pieces:
        for g, s in pc["cons"]:
            if not all(g.has(v) for v in grp[:2]) and n == 2:
                pass
            cg = _coeffs_groupe(g, grp)
            if cg is None or cg["cross"]:
                continue
            sq = cg["sq"]
            if all(c.is_number and c != 0 for c in sq) and all(np.sign(float(c)) == np.sign(float(sq[0])) for c in sq):
                if ref is None:
                    ref = (sq, cg["lin"], pc)
                break
        if ref is not None:
            break
    if ref is None:
        # pas de contrainte quadratique : on tente les coordonnées centrées si f est radiale en grp
        try:
            rng = np.random.RandomState(7)
            radial = True
            for pc in pieces:
                fn = _lamb(pc["f"], list(grp))
                if fn is None or not pc["f"].has(*grp) or sum(1 for v in grp if pc["f"].has(v)) < n:
                    radial = False
                    break
                for r in (0.7, 1.9):
                    vals = []
                    for _ in range(4):
                        d = rng.normal(size=n)
                        d = d / np.linalg.norm(d) * r
                        with np.errstate(all="ignore"):
                            vals.append(complex(fn(*d)))
                    if not all(np.isfinite(t) for t in vals) or max(abs(t - vals[0]) for t in vals) > 1e-9 * (1 + abs(vals[0])):
                        radial = False
                        break
                if not radial:
                    break
            if radial:
                ref = ([sp.Integer(1)] * n, [sp.Integer(0)] * n, pieces[0])
        except Exception:
            ref = None
    if ref is None:
        return None
    sq0, lin0, _ = ref
    centre = [sp.simplify(-lin0[i] / (2 * sq0[i])) for i in range(n)]
    if not all(c.is_number for c in centre):
        return None
    sc = [sp.sqrt(sp.Abs(c)) for c in sq0]
    # nouvelles variables
    if n == 2:
        rr, th = _noms_libres(V, ["r", "theta"])
        nouv = [th, rr]
        coord = [centre[0] + rr * sp.cos(th) / sc[0], centre[1] + rr * sp.sin(th) / sc[1]]
        jac = rr / (sc[0] * sc[1])
        angles = [th]
        rayon = rr
    else:
        rr, th, ph = _noms_libres(V, ["rho", "theta", "varphi"])
        nouv = [th, ph, rr]
        coord = [centre[0] + rr * sp.sin(ph) * sp.cos(th) / sc[0], centre[1] + rr * sp.sin(ph) * sp.sin(th) / sc[1], centre[2] + rr * sp.cos(ph) / sc[2]]
        jac = rr ** 2 * sp.sin(ph) / (sc[0] * sc[1] * sc[2])
        angles = [th, ph]
        rayon = rr
    sub = dict(zip(grp, coord))
    autres = [v for v in V if v not in grp]
    nouvelles_pieces = []
    for pc in pieces:
        cons_n = []
        arcs_th = []
        borne_ph = [(sp.Integer(0), sp.pi)]
        radial_vu = False
        for g, s in pc["cons"]:
            cg = _coeffs_groupe(g, grp)
            touche = any(g.has(v) for v in grp)
            if not touche:
                cons_n.append((g, s))
                continue
            # contrainte radiale (même centre, même rapport d'axes)
            if cg is not None and not cg["cross"] and all(c.is_number for c in cg["sq"]) and all(c != 0 for c in cg["sq"]):
                rap = [sp.simplify(cg["sq"][i] / sq0[i]) for i in range(n)]
                memec = all(sp.simplify(rap[i] - rap[0]) == 0 for i in range(n)) and all(
                    sp.simplify(cg["lin"][i] + 2 * cg["sq"][i] * centre[i]) == 0 for i in range(n))
                if memec:
                    radial_vu = True
                    e = sp.simplify(sp.trigsimp(g.subs(sub, simultaneous=True)))
                    cons_n.append((e, s))
                    continue
            # demi-espace / demi-plan passant par le centre : combinaison linéaire des (v_i - c_i)
            if cg is not None and all(c == 0 for c in cg["sq"]) and not cg["cross"]:
                delta = sp.simplify(cg["cst"] + sum(cg["lin"][i] * centre[i] for i in range(n)))
                coef = [cg["lin"][i] / 1 for i in range(n)]
                # coefficients de u_i = (v_i - c_i) : lin_i ; coefficient devant r : lin_i / sc_i
                if delta == 0 and all(c.is_number for c in coef) and not any(g.has(v) for v in autres):
                    if n == 2:
                        al, be = coef[0] / sc[0], coef[1] / sc[1]
                        # g = al r cos + be r sin <= 0  <=> -(al cos + be sin) >= 0
                        arcs_th.append(_arc_demiplan(al, be, -1))
                        continue
                    else:
                        al, be, ga = coef[0] / sc[0], coef[1] / sc[1], coef[2] / sc[2]
                        if ga == 0 and (al != 0 or be != 0):
                            arcs_th.append(_arc_demiplan(al, be, -1))
                            continue
                        if al == 0 and be == 0 and ga != 0:
                            # ga r cos(phi) <= 0
                            if ga > 0:
                                borne_ph.append((sp.pi / 2, sp.pi))
                            else:
                                borne_ph.append((sp.Integer(0), sp.pi / 2))
                            continue
                        return None
            # autre contrainte : substituée telle quelle (dans le cas cylindrique par exemple)
            e = sp.simplify(sp.trigsimp(g.subs(sub, simultaneous=True)))
            cons_n.append((e, s))
        if not radial_vu and pc is pieces[0]:
            pass
        arc = _inter_arcs(arcs_th) if n >= 2 else None
        if arcs_th and arc is None:
            continue
        lo_t, hi_t = arc if arcs_th else (sp.Integer(0), 2 * sp.pi)
        cons_n.append((lo_t - th, False))
        cons_n.append((th - hi_t, False))
        cons_n.append((-rayon, False))
        if n == 3:
            lo_p = max([a for a, b in borne_ph], key=lambda t: float(t))
            hi_p = min([b for a, b in borne_ph], key=lambda t: float(t))
            cons_n.append((lo_p - ph, False))
            cons_n.append((ph - hi_p, False))
        fn = sp.simplify(sp.trigsimp(pc["f"].subs(sub, simultaneous=True) * jac))
        nouvelles_pieces.append(dict(f=fn, cons=cons_n))
    if not nouvelles_pieces:
        return None
    Vn = list(nouv) + autres
    kind = "sphérique" if n == 3 else ("polaire" if len(V) == 2 else "cylindrique")
    gen = any(c != 0 for c in centre) or any(s_ != 1 for s_ in sc)
    return dict(kind=kind, nom=kind, generalise=gen, grp=grp, sub=sub, jac=sp.simplify(jac), centre=centre, sc=sc, V=Vn, pieces=nouvelles_pieces, nouv=nouv, rayon=rayon, angles=angles)


# -------------------------------------------------------------
# Résolution d'une route (coordonnées données) : ordres, cellules, intégrales itérées
# -------------------------------------------------------------
def _resoudre(pieces, V, budget=30.0, max_ordres=4):
    t0 = time.time()
    ordres, ech = _ordres_possibles(pieces, V, budget=min(budget, 15.0))
    res = dict(ok=False, ordres=[], echec_domaine=ech, choix=None, autre=None, cells_any=(ordres[0][2] if ordres else None))
    for sc, p, cells in ordres:
        if time.time() - t0 > budget:
            break
        total = sp.Integer(0)
        resultats = []
        echec = None
        for f_, c in cells:
            val, steps = _calcul_cellule(f_, c, deadline=t0 + budget)
            if val is None:
                echec = dict(cell=c, f=f_, steps=steps)
                break
            resultats.append((f_, c, val, steps))
            total = total + val
        if echec is not None:
            res["ordres"].append(dict(ordre=p, ok=False, echec=echec, cells=cells))
            continue
        try:
            total = sp.simplify(total)
        except Exception:
            pass
        if total.has(sp.nan, sp.zoo):
            res["ordres"].append(dict(ordre=p, ok=False, echec=None, cells=cells))
            continue
        entree = dict(ordre=p, ok=True, cells=cells, resultats=resultats, total=total, score=sc)
        res["ordres"].append(entree)
        if res["choix"] is None:
            res["choix"] = entree
            res["ok"] = True
        elif res["autre"] is None:
            res["autre"] = entree
            break
        if res["choix"] is entree and time.time() - t0 > 4.0:
            break   # le contrôle par un second ordre est facultatif : on ne le paie pas s'il est coûteux
        if time.time() - t0 > budget:
            break
    return res


# -------------------------------------------------------------
# Signe, symétries, séparabilité
# -------------------------------------------------------------
def _echant_global(cells, m=40):
    pts = []
    for _, c in cells[:12]:
        pts += _echant_cellule(c, m=m, seed=11)
    return pts


def _evals_f(f, V, pts):
    fn = _lamb(f, V)
    vals = []
    for p in pts:
        try:
            v = complex(fn(*[p.get(x, 0.0) for x in V]))
            if abs(v.imag) < 1e-9 and np.isfinite(v.real):
                vals.append(v.real)
        except Exception:
            pass
    return vals


def _signe_f(f, V, cells_cart):
    pts = _echant_global(cells_cart)
    vals = _evals_f(f, V, pts)
    if not vals:
        return None
    if all(abs(v) < 1e-14 for v in vals):
        return "0"
    if all(v >= -1e-12 for v in vals):
        return "+"
    if all(v <= 1e-12 for v in vals):
        return "-"
    return "mixte"


def _raison_signe(f, V, pts, positif=True):
    """Phrase brève justifiant f >= 0 (ou <= 0) sur D, par facteurs."""
    try:
        c0, facs = sp.factor_list(sp.powsimp(f))
    except Exception:
        return None
    morceaux = []
    for base, ex in facs:
        if isinstance(base, sp.exp) or (base.is_Pow and base.base == sp.E):
            morceaux.append("$" + sp.latex(base) + ">0$")
        elif ex % 2 == 0:
            morceaux.append("$" + (r"\left(" + sp.latex(base) + r"\right)" if base.is_Add else sp.latex(base)) + "^{" + str(ex) + "}\\ge0$")
        else:
            vals = _evals_f(base, V, pts)
            if vals and all(v >= -1e-12 for v in vals):
                morceaux.append("$" + sp.latex(base) + "\\ge0$ sur $D$")
            elif vals and all(v <= 1e-12 for v in vals):
                morceaux.append("$" + sp.latex(base) + "\\le0$ sur $D$")
            else:
                return None
    if not morceaux:
        return None
    return ", ".join(morceaux)


def _symetrique(pieces, V, v, nb=3000):
    rng = np.random.RandomState(5)
    j = V.index(v)
    ok = True
    for R in (1.5, 4.0):
        P = rng.uniform(-R, R, size=(nb, len(V)))
        Q = P.copy()
        Q[:, j] = -Q[:, j]
        a = np.zeros(nb, dtype=bool)
        b = np.zeros(nb, dtype=bool)
        for pc in pieces:
            a |= _masque(pc["cons"], V, P)
            b |= _masque(pc["cons"], V, Q)
        if np.any(a != b):
            return False
        if not a.any():
            return False
    return True


def _separable(f, V):
    """Liste de (facteur, variable) si f = produit de fonctions de variables séparées, sinon None."""
    try:
        ff = sp.factor(f)
    except Exception:
        ff = f
    groupes = {}
    cst = sp.Integer(1)
    for fac in sp.Mul.make_args(ff):
        vs = [v for v in V if fac.has(v)]
        if not vs:
            cst *= fac
        elif len(vs) == 1:
            groupes[vs[0]] = groupes.get(vs[0], sp.Integer(1)) * fac
        else:
            return None
    if len(groupes) < 2:
        return None
    return cst, groupes


def _est_pave(cells):
    if len(cells) != 1:
        return None
    _, c = cells[0]
    bornes = {}
    for v, lo, hi in c:
        if (lo not in (sp.oo, -sp.oo) and not lo.is_number) or (hi not in (sp.oo, -sp.oo) and not hi.is_number):
            return None
        bornes[v] = (lo, hi)
    return bornes


def _f_usuelle_bornee(f, V):
    """f est un polynôme, exp, sin, cos, sinh, cosh, atan... : continue sur tout ℝ^k."""
    ok_types = (sp.exp, sp.sin, sp.cos, sp.sinh, sp.cosh, sp.atan, sp.tanh)
    for a in sp.preorder_traversal(f):
        if isinstance(a, sp.Pow):
            if not (a.exp.is_Integer and a.exp >= 0) and a.base != sp.E:
                return False
        elif isinstance(a, sp.Function) and not isinstance(a, ok_types):
            return False
    return True


def _borne_domaine(cells):
    """True si toutes les bornes numériques de chaque variable dans toutes les cellules sont finies."""
    for _, c in cells:
        for v, lo, hi in c:
            if lo == -sp.oo or hi == sp.oo:
                return False
    return True


# -------------------------------------------------------------
# Écriture LaTeX
# -------------------------------------------------------------
def _lb(b):
    return _L._borne_lat(b)


def _lat_vars(V):
    return "(" + ",".join(sp.latex(v) for v in V) + ")"


def _lat_conds_cell(cell):
    morceaux = []
    for v, lo, hi in cell:
        sv = sp.latex(v)
        if lo == -sp.oo and hi == sp.oo:
            morceaux.append(sv + r"\in\mathbb{R}")
        elif lo == -sp.oo:
            morceaux.append(sv + r"\le " + _lb(hi))
        elif hi == sp.oo:
            morceaux.append(sv + r"\ge " + _lb(lo))
        else:
            morceaux.append(_lb(lo) + r"\le " + sv + r"\le " + _lb(hi))
    return r",\ ".join(morceaux)


def _lat_ens(V, conds):
    k = len(V)
    if not conds:
        return r"\mathbb{R}^{%d}" % k
    return r"\left\{" + _lat_vars(V) + r"\in\mathbb{R}^{%d}\ :\ " % k + r",\ ".join(conds) + r"\right\}"


def _lat_ens_cell(V, cell):
    return r"\left\{" + _lat_vars(V) + r"\ :\ " + _lat_conds_cell(cell) + r"\right\}"


def _dx(vars_):
    return r"\,".join("d" + sp.latex(v) for v in vars_)


def _val_lat(v):
    if v == sp.oo:
        return r"+\infty"
    return sp.latex(v)


# -------------------------------------------------------------
# Contrôle numérique (indicatif) par intégration itérée
# -------------------------------------------------------------
def _controle_numerique(cells, V):
    try:
        from scipy import integrate as si
    except Exception:
        return None
    total = 0.0
    for f_, c in cells:
        ordre = [x[0] for x in c]
        k = len(ordre)
        fn = _lamb(f_, ordre)
        if fn is None:
            return None
        bornes = []
        for i, (v, lo, hi) in enumerate(c):
            outers = ordre[:i]
            lf = (lambda *a: -np.inf) if lo == -sp.oo else _lamb(lo, outers) if outers else (lambda *a, _l=lo: float(_l))
            hf = (lambda *a: np.inf) if hi == sp.oo else _lamb(hi, outers) if outers else (lambda *a, _h=hi: float(_h))
            bornes.append((lf, hf))

        # nquad : x0 = variable la plus intérieure ; les bornes reçoivent les variables plus extérieures
        def func(*args):
            vals = list(reversed(args))
            return float(fn(*vals))

        ranges = []
        for i in range(k - 1, -1, -1):
            lf, hf = bornes[i]
            def mk(lf=lf, hf=hf):
                def r(*a):
                    # a : variables extérieures dans l'ordre nquad (de la plus intérieure à la plus extérieure)
                    vals = list(reversed(a))
                    return [float(lf(*vals)), float(hf(*vals))]
                return r
            ranges.append(mk())
        # le niveau le plus extérieur reçoit une liste constante
        e_lo, e_hi = bornes[0]
        ranges[-1] = [float(e_lo()), float(e_hi())]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = _delai(lambda: si.nquad(func, ranges, opts=dict(limit=40, epsabs=1e-7, epsrel=1e-7))[0], 25 if k <= 2 else (10 if k == 3 else 5))
        if r is None:
            return None
        total += r
    return total


# -------------------------------------------------------------
# Graphes
# -------------------------------------------------------------
def _fenetre_points(pts, V, marge=0.08):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ext = (maxs - mins)
    ext[ext == 0] = 1.0
    return mins - marge * ext, maxs + marge * ext


def _graphes(f, pieces, V, cells_cart, borne):
    k = len(V)
    if k < 2 or k > 3:
        _P(_T("Pas de représentation graphique en dimension %d." % k))
        return
    pts = np.vstack([_points_dans(pc["cons"], V, n=1200) for pc in pieces if len(_points_dans(pc["cons"], V, n=50)) > 0] or [np.zeros((0, k))])
    if len(pts) < 5:
        return
    lo, hi = _fenetre_points(pts, V)
    fn = _lamb(f, V)
    if k == 2:
        n = 220
        gx = np.linspace(lo[0], hi[0], n)
        gy = np.linspace(lo[1], hi[1], n)
        X, Y = np.meshgrid(gx, gy)
        P = np.column_stack([X.ravel(), Y.ravel()])
        m = np.zeros(len(P), dtype=bool)
        for pc in pieces:
            m |= _masque(pc["cons"], V, P)
        M = m.reshape(X.shape)
        fig = plt.figure(figsize=(11, 4.6))
        ax = fig.add_subplot(1, 2, 1)
        ax.contourf(X, Y, M.astype(float), levels=[0.5, 1.5], colors=["#9ecae1"], alpha=0.85)
        ax.contour(X, Y, M.astype(float), levels=[0.5], colors=["#08519c"], linewidths=1.4)
        # tranches d'intégration (cellules)
        for f_, c in cells_cart[:6]:
            (v0, a0, b0), (v1, a1, b1) = c[0], c[1]
            ctx0 = [a0, b0]
            try:
                A = float(a0) if a0 != -sp.oo else lo[V.index(v0)]
                B = float(b0) if b0 != sp.oo else hi[V.index(v0)]
            except Exception:
                continue
            for t in np.linspace(A, B, 9)[1:-1]:
                try:
                    u0 = float(a1.subs(v0, t)) if a1 != -sp.oo else lo[V.index(v1)]
                    u1 = float(b1.subs(v0, t)) if b1 != sp.oo else hi[V.index(v1)]
                except Exception:
                    continue
                if V.index(v0) == 0:
                    ax.plot([t, t], [u0, u1], color="#d94801", lw=0.7, alpha=0.8)
                else:
                    ax.plot([u0, u1], [t, t], color="#d94801", lw=0.7, alpha=0.8)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xlabel(sp.latex(V[0]).join(["$", "$"]))
        ax.set_ylabel(sp.latex(V[1]).join(["$", "$"]))
        ax.set_title("Domaine $D$ et tranches d'intégration")
        ax.set_aspect("auto")
        ax.grid(alpha=0.25)
        ax3 = fig.add_subplot(1, 2, 2, projection="3d")
        with np.errstate(all="ignore"):
            Z = np.asarray(fn(X, Y), dtype=float) * np.ones(X.shape)
        Z = np.where(M & np.isfinite(Z), Z, np.nan)
        zs = Z[np.isfinite(Z)]
        if zs.size:
            cap = np.percentile(np.abs(zs), 98) * 1.2 + 1e-9
            Z = np.where(np.abs(Z) <= cap, Z, np.nan)
        ax3.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
        ax3.set_title("Surface $z=f$ au-dessus de $D$")
        ax3.set_xlabel(sp.latex(V[0]).join(["$", "$"]))
        ax3.set_ylabel(sp.latex(V[1]).join(["$", "$"]))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        sel = pts[_RNG.permutation(len(pts))[:2500]]
        with np.errstate(all="ignore"):
            vals = np.asarray(fn(sel[:, 0], sel[:, 1], sel[:, 2]), dtype=float) * np.ones(len(sel))
        fig = plt.figure(figsize=(7.5, 5.8))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        sc = ax.scatter(sel[:, 0], sel[:, 1], sel[:, 2], c=vals, s=6, cmap="viridis", alpha=0.7)
        fig.colorbar(sc, ax=ax, shrink=0.6, label="$f$")
        ax.set_xlabel(sp.latex(V[0]).join(["$", "$"]))
        ax.set_ylabel(sp.latex(V[1]).join(["$", "$"]))
        ax.set_zlabel(sp.latex(V[2]).join(["$", "$"]))
        ax.set_title("Points de $D$ colorés par la valeur de $f$")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# -------------------------------------------------------------
# Rédaction
# -------------------------------------------------------------
def _signe_points(f, V, pieces):
    pts = []
    for pc in pieces:
        P = _points_dans(pc["cons"], V, n=800)
        if len(P):
            fn = _lamb(pc["f"], V)
            with np.errstate(all="ignore"):
                try:
                    vals = np.asarray(fn(*[P[:, j] for j in range(len(V))]), dtype=complex) * np.ones(len(P))
                except Exception:
                    continue
            vals = vals[np.isfinite(vals)]
            vals = vals[np.abs(vals.imag) < 1e-9].real
            pts.append((P, vals))
    return pts


def _classe_signe(pts):
    vals = np.concatenate([v for _, v in pts]) if pts else np.array([])
    if vals.size == 0:
        return None
    if np.all(np.abs(vals) < 1e-14):
        return "0"
    if np.all(vals >= -1e-12):
        return "+"
    if np.all(vals <= 1e-12):
        return "-"
    return "mixte"


def _txt_ordre(p):
    return "(" + ",".join(sp.latex(v) for v in p) + ")"


def _integrabilite(f, V, pieces, route, cart):
    """Retourne (statut, lignes) : True / False / None pour f ∈ L¹(D)."""
    lignes = []
    cells = route["choix"]["cells"]
    D_borne = _borne_domaine(cells)
    if D_borne and _f_usuelle_bornee(f, V):
        lignes.append(_T("Le domaine $D$ est borné, donc de mesure finie, et $f$ est continue sur $\\mathbb{R}^{%d}$ (fonction usuelle) : elle est bornée sur $D$ par une constante $M$, d'où $\\int_D|f|\\,d\\lambda\\le M\\,\\lambda(D)<+\\infty$." % len(V)))
        return True, lignes
    # variables séparées sur un pavé
    if cart is not None and cart.get("choix") is not None:
        sep = _separable(f, V)
        pave = _est_pave(cart["choix"]["cells"])
        if sep is not None and pave is not None:
            cst, groupes = sep
            lignes.append(_T("Sur le pavé $D=" + r"\times".join(_L._iv(*pave[v], False, False) if (pave[v][0] not in (sp.oo, -sp.oo) and pave[v][1] not in (sp.oo, -sp.oo)) else _L._iv(pave[v][0], pave[v][1], True, True) for v in V) + "$, $f$ est à variables séparées : " + "$f=" + sp.latex(cst) + r"\cdot " + r"\cdot ".join(sp.latex(groupes[v]) for v in V if v in groupes) + "$. Par Tonelli, $\\int_D|f|=|" + sp.latex(cst) + "|\\prod_i\\int|g_i|$ : il suffit d'étudier chaque facteur séparément."))
            tous = True
            un_inf = False
            for v in V:
                if v not in groupes:
                    continue
                lo, hi = pave[v]
                vv, lg = _L._integrable_g(groupes[v], v, lo, hi, "g")
                lignes.append(_T("En la variable $" + sp.latex(v) + "$ : $g=" + sp.latex(groupes[v]) + "$."))
                lignes += lg
                if vv is not True:
                    tous = False
                    if vv is False:
                        un_inf = True
            if tous:
                lignes.append(_T("Chaque facteur est intégrable : $\\int_D|f|<+\\infty$."))
                return True, lignes
            if un_inf and cst != 0:
                lignes.append(_T("Un facteur n'est pas intégrable et les autres ne sont pas identiquement nuls : $\\int_D|f|=+\\infty$."))
                return False, lignes
    # signe constant sur chaque cellule
    tot = sp.Integer(0)
    ok = True
    for f_, c, val, steps in route["choix"]["resultats"]:
        pts = _echant_cellule(c, m=60, seed=21)
        vals = _evals_f(f_, [x[0] for x in c], pts)
        if not vals:
            ok = False
            break
        if all(v >= -1e-12 for v in vals):
            tot += val
        elif all(v <= 1e-12 for v in vals):
            tot += -val
        else:
            ok = False
            break
    if ok:
        tot = sp.simplify(tot)
        if tot == sp.oo:
            lignes.append(_T("Sur chaque domaine de calcul $f$ garde un signe constant : $\\int_D|f|=\\sum\\left|\\int f\\right|=+\\infty$."))
            return False, lignes
        lignes.append(_T("Sur chaque domaine de calcul $f$ garde un signe constant, donc $|f|=\\pm f$ et le théorème de Tonelli donne $\\int_D|f|\\,d\\lambda=" + sp.latex(tot) + "<+\\infty$."))
        return True, lignes
    return None, lignes


def _phrase_ordre(cart, choix):
    """Explique le changement d'ordre quand le premier ordre essayé échoue."""
    if cart is None or not cart.get("ordres"):
        return None
    prem = cart["ordres"][0]
    if prem["ok"] or choix is None or tuple(prem["ordre"]) == tuple(choix["ordre"]):
        return None
    ech = prem.get("echec")
    if not ech:
        return None
    last = [s for s in ech["steps"] if s.get("echec")]
    if not last:
        return None
    s = last[-1]
    return _T("Dans l'ordre " + "$" + _txt_ordre(prem["ordre"]) + "$, l'intégrale $\\int_{" + _lb(s["lo"]) + "}^{" + _lb(s["hi"]) + "} " + sp.latex(s["g"]) + "\\,d" + sp.latex(s["v"]) + "$ ne s'exprime pas à l'aide de fonctions usuelles. Les hypothèses du théorème de Fubini–Tonelli étant satisfaites, on intègre dans l'autre ordre $" + _txt_ordre(choix["ordre"]) + "$.")


_NOTE_ERF = []


def choix_total_tmp(route):
    return route["choix"]["total"]


def _est_radial(chg):
    """f∘Φ ne dépend d'aucun angle (le jacobien, lui, peut en dépendre)."""
    try:
        return all(not sp.simplify(pc["f"] / chg["jac"]).has(*chg["angles"]) for pc in chg["pieces"])
    except Exception:
        return False


def _a_racine(b):
    for a in sp.preorder_traversal(b):
        if isinstance(a, sp.Pow) and a.exp.is_Rational and not a.exp.is_Integer:
            return True
    return False


def _narrer_cellules(resultats, V, nom_cells):
    multiple = len(resultats) > 1
    for i, (f_, c, val, steps) in enumerate(resultats):
        if multiple:
            _P(_T("Sur $D_{%d}=" % (i + 1) + _lat_ens_cell(V if V else [x[0] for x in c], c) + "$ :"))
        for j, stp in enumerate(steps):
            v, lo, hi, g, F, vl = stp["v"], stp["lo"], stp["hi"], stp["g"], stp["F"], stp["val"]
            sv = sp.latex(v)
            outers = stp["outers"]
            if j == 0:
                intro = _L._V("mi0", "On commence par intégrer en $%s$, les autres variables étant fixées :" % sv,
                              "Intégrons d'abord par rapport à $%s$ (les autres variables sont des paramètres) :" % sv,
                              "Fixons les autres variables et intégrons en $%s$ :" % sv)
            else:
                par = ",".join(sp.latex(o) for o in outers) if outers else ""
                intro = _L._V("mi1", "Il reste à intégrer en $%s$ :" % sv,
                              "Puis, en $%s$ :" % sv,
                              "On poursuit avec la variable $%s$ :" % sv)
            _P(_T(intro))
            integrale = r"\int_{" + _lb(lo) + "}^{" + _lb(hi) + "} " + sp.latex(g) + r"\,d" + sv
            if F is not None:
                lg = []
                try:
                    lg = _L._narrer_primitive(g, v, F) or []
                except Exception:
                    lg = []
                for l_ in lg:
                    _P(l_)
                if F.has(sp.erf) and not any("erf" in x for x in _NOTE_ERF):
                    _NOTE_ERF.append("erf")
                    _P(_T("Ici apparaît la fonction d'erreur : $\\operatorname{erf}(t)=\\frac{2}{\\sqrt{\\pi}}\\int_0^{t}e^{-s^{2}}\\,ds$, primitive non élémentaire de $e^{-t^{2}}$."))
                Fl = sp.latex(F)
                _P(integrale + r"=\left[" + Fl + r"\right]_{" + _lb(lo) + "}^{" + _lb(hi) + "}=" + _val_lat(vl))
            else:
                if lo == -sp.oo and hi == sp.oo and g.has(sp.exp):
                    _P(_T("On reconnaît une intégrale de type Gauss : $\\int_{\\mathbb{R}}e^{-a t^{2}+b t}\\,dt=\\sqrt{\\pi/a}\\,e^{b^{2}/(4a)}$ pour $a>0$."))
                _P(integrale + "=" + _val_lat(vl))
        if multiple:
            _P(_T("Donc l'intégrale sur $D_{%d}$ vaut $%s$." % (i + 1, _val_lat(val))))


def _decrit_changement(chg, pieces_orig, V, k):
    grp = chg["grp"]
    sub = chg["sub"]
    nom = {2: "polaires", 3: "sphériques"}[len(grp)]
    if len(grp) == 2 and k > 2:
        nom = "cylindriques"
    lig = []
    for v, e in zip(grp, [sub[g] for g in grp]):
        lig.append(sp.latex(v) + "=" + sp.latex(e))
    return nom, r",\quad ".join(lig)


# -------------------------------------------------------------
# Étude complète
# -------------------------------------------------------------
def _etude_interne(fstr, dstr, vstr, loc, N=None):
    _L._VUES.clear()
    _L._ROT.clear()
    _NOTE_ERF.clear()
    _push()
    try:
        f, cons, V = _analyser(fstr, dstr, vstr, loc)
        aff = list(_AFF)
        if aff and all(re.fullmatch(r"-\\infty < [A-Za-z_]\w* < \\infty", a_) for a_ in aff):
            aff = []
        pieces = _scinder(f, cons, V)
    except _NonSupporte as e:
        _rouge(str(e))
        return
    except Exception as e:
        _rouge("Entrée non comprise (%s)." % str(e)[:120])
        return
    k = len(V)

    # ---------------- Données ----------------
    _titre("Données")
    _P(_T("On étudie l'intégrale multiple $I=\\int_D f\\,d\\lambda_{%d}=\\int\\!\\!\\cdots\\!\\!\\int_D f%s\\,%s$ avec $f%s=%s$ et $D=%s$." % (k, _lat_vars(V), _dx(V), _lat_vars(V), sp.latex(f), _lat_ens(V, aff))))

    # ---------------- Résolution ----------------
    chg = None
    rep = None
    try:
        chg = _essai_changement(f, pieces, V)
    except Exception:
        chg = None
    cart = None
    if chg is not None and _est_radial(chg):
        # situation radiale : les coordonnées polaires / cylindriques / sphériques s'imposent, on ne cherche pas plus loin
        rep = _resoudre(chg["pieces"], chg["V"], budget=25.0)
        if rep["ok"]:
            try:
                _o, _e = _ordres_possibles(pieces, V, budget=6.0, maximum=1)
            except Exception:
                _o, _e = [], None
            cart = dict(ok=False, ordres=[], echec_domaine=_e, choix=None, autre=None, cells_any=(_o[0][2] if _o else None))
    if cart is None:
        cart = _resoudre(pieces, V, budget=30.0)
        if chg is not None and rep is None:
            rep = _resoudre(chg["pieces"], chg["V"], budget=25.0)

    route, nom_route = None, "cartésien"
    if cart["ok"] and not (rep and rep["ok"]):
        route = cart
    elif rep and rep["ok"] and not cart["ok"]:
        route, nom_route = rep, "changement"
    elif cart["ok"] and rep and rep["ok"]:
        radial = _est_radial(chg)
        lourd = len(cart["choix"]["cells"]) > 1 or any(_a_racine(b) for _c, c in cart["choix"]["cells"] for _v, lo, hi in c for b in (lo, hi) if b not in (sp.oo, -sp.oo))
        if radial or lourd:
            route, nom_route = rep, "changement"
        else:
            route = cart
    # changements généraux (linéaires, hyperboliques) : seulement s'ils simplifient ou si rien d'autre ne marche
    if nom_route != "changement":
        try:
            cpl_cart = _couplees(pieces, V)
            if (not cart["ok"]) or cpl_cart > 0:
                cands = _candidats_lineaires(f, pieces, V) + _candidats_hyperboliques(f, pieces, V)
                cands.sort(key=lambda c_: c_["pre"])
                cout_cart = _cout_route(cart)
                meilleur = None
                t_gen = time.time()
                for c_ in cands[:4]:
                    if time.time() - t_gen > 40:
                        break
                    if cart["ok"] and not (c_["couplees"] < cpl_cart):
                        continue
                    r_ = _resoudre(c_["pieces"], c_["V"], budget=15.0)
                    co_ = _cout_route(r_)
                    if r_["ok"] and (meilleur is None or (co_, c_["pre"]) < (meilleur[0], meilleur[1]["pre"])):
                        meilleur = (co_, c_, r_)
                        if c_["couplees"] == 0 and co_ <= 8:
                            break
                if meilleur is not None and ((not cart["ok"]) or meilleur[0] < 0.75 * cout_cart):
                    chg, rep = meilleur[1], meilleur[2]
                    route, nom_route = rep, "changement"
        except Exception:
            pass

    # ---------------- Le domaine ----------------
    _titre("Le domaine")
    _P(_T("$D$ est défini par des inégalités portant sur des fonctions continues : c'est un borélien de $\\mathbb{R}^{%d}$, donc une partie mesurable. %s" % (k, "Quant à $f$, c'est une fonction continue sur $\\mathbb{R}^{%d}$, donc mesurable." % k if _f_usuelle_bornee(f, V) else "Quant à $f$, elle est continue là où elle est définie (composée de fonctions usuelles), donc mesurable.")))
    cells_desc = None
    if cart.get("cells_any"):
        cells_desc = cart["choix"]["cells"] if cart["ok"] else cart["cells_any"]
    if cells_desc:
        c0 = cells_desc
        if len(c0) == 1:
            _P(_T("Décrit comme un domaine emboîté, $D=" + _lat_ens_cell(V, c0[0][1]) + "$."))
        else:
            _P(_T("On découpe $D$ en %d domaines emboîtés, disjoints à un ensemble négligeable près : $D=%s$." % (len(c0), r"\cup ".join("D_{%d}" % (i + 1) for i in range(len(c0))))))
            for i, (_, c) in enumerate(c0):
                _P("D_{%d}=" % (i + 1) + _lat_ens_cell(V, c))
        borne = _borne_domaine(c0)
        _P(_T("Le domaine est %s." % ("borné : sa mesure est finie" if borne else "non borné : sa mesure est infinie, et l'intégrabilité de $f$ devra être étudiée avec soin")))
    elif cart.get("echec_domaine"):
        _P(_T("Description du domaine : " + str(cart["echec_domaine"])))

    if route is None:
        _titre("Calcul")
        _P(_T("Aucun ordre d'intégration ne conduit à des intégrales itérées exprimables avec des fonctions usuelles, ni aucun changement de variables reconnu : le calcul exact n'a pas pu être mené."))
        try:
            obst = None
            for o_ in (cart.get("ordres") or []):
                ech_ = o_.get("echec")
                if ech_:
                    for s_ in ech_["steps"]:
                        if s_.get("echec"):
                            obst = s_
                            break
                if obst:
                    break
            if obst is not None:
                _P(_T("Obstacle : $\\int_{%s}^{%s}%s\\,d%s$ n'admet pas de primitive élémentaire exploitable (les fonctions spéciales comme la fonction d'erreur sont exclues)." % (_lb(obst["lo"]), _lb(obst["hi"]), "\\left(" + sp.latex(obst["g"]) + "\\right)" if obst["g"].is_Add else sp.latex(obst["g"]), sp.latex(obst["v"]))))
        except Exception:
            pass
        if cells_desc:
            val = _controle_numerique(cells_desc, V)
            if val is not None:
                _P(_T("À titre de simple contrôle numérique (sans valeur de preuve), $I\\approx %.6g$." % val))
        _titre("Graphes")
        _graphes(f, pieces, V, cells_desc or [], None)
        return

    choix = route["choix"]
    f_route_cells = choix["cells"]

    # ---------------- Signe / théorème ----------------
    _titre("Théorème applicable")
    pts_sig = _signe_points(f, V, pieces)
    signe = _classe_signe(pts_sig)
    pts_all = np.vstack([P for P, _ in pts_sig]) if pts_sig else np.zeros((0, k))
    pts_dicts = [dict(zip(V, row)) for row in pts_all[:200]]
    statut_int = None
    lignes_int = []
    if signe in ("+", "0", "-"):
        raison = _raison_signe(f, V, pts_dicts, signe == "+")
        pos = signe != "-"
        g_txt = "f" if pos else "-f"
        if signe == "0":
            _P(_T("La fonction $f$ est nulle sur $D$ : $I=0$."))
        else:
            _P(_T("Pour tout $%s\\in D$, $%sf%s\\ge0$%s." % (_lat_vars(V), "" if pos else "-", _lat_vars(V), (" (" + raison + ")") if raison else " (étude du signe de $f$ sur $D$)")))
        _P(_T(_L._V("tonelli", "Comme %s est mesurable et positive, le théorème de Fubini–Tonelli s'applique sans autre hypothèse : $\\int_D %s$ est égale à l'intégrale itérée, dans l'ordre que l'on choisit, avec une valeur dans $[0,+\\infty]$." % ("$" + g_txt + "$", g_txt),
                          "La fonction $%s$ étant mesurable et à valeurs positives, Fubini–Tonelli permet de calculer $\\int_D %s$ par intégrales itérées, dans n'importe quel ordre, sans vérifier d'intégrabilité (le résultat peut être $+\\infty$)." % (g_txt, g_txt))))
        statut_int = True
    else:
        _P(_T("La fonction $f$ change de signe sur $D$ : le théorème de Tonelli ne suffit plus. On l'applique d'abord à $|f|\\ge0$ pour décider si $f\\in L^{1}(D)$, puis, si c'est le cas, le théorème de Fubini autorise les intégrales itérées."))
        statut_int, lignes_int = _integrabilite(f, V, pieces, route, cart if cart["ok"] else None)
        if statut_int is None:
            try:
                pa = _scinder(sp.Abs(f), cons, V)
                ra = _resoudre(pa, V, budget=25.0)
            except Exception:
                ra = None
            if ra and ra["ok"]:
                ta = ra["choix"]["total"]
                lignes_int.append(_T("Le calcul de $\\int_D|f|\\,d\\lambda$ s'obtient en découpant $D$ selon le signe de $f$ (là où $f\\ge0$, $|f|=f$ ; ailleurs, $|f|=-f$) et en appliquant Tonelli à $|f|\\ge0$ sur chaque morceau, dans l'ordre $%s$ : on trouve $\\int_D|f|\\,d\\lambda=%s$." % (_txt_ordre(ra["choix"]["ordre"]), _val_lat(ta))))
                if ta == sp.oo:
                    statut_int = False
                elif ta.is_finite:
                    statut_int = True
        for l_ in lignes_int:
            _P(l_)
        if statut_int is None:
            au_ = route.get("autre")
            try:
                if au_ is not None and au_["total"] != choix_total_tmp(route) and not (sp.simplify(au_["total"] - choix_total_tmp(route)) == 0):
                    _P(_T("Les deux intégrales itérées, dans les ordres $%s$ et $%s$, existent mais n'ont pas la même valeur ($%s$ et $%s$). Si $f$ était intégrable sur $D$, le théorème de Fubini imposerait l'égalité des deux : par contraposée, $f\\notin L^{1}(D)$ et $\\int_D|f|\\,d\\lambda=+\\infty$." % (_txt_ordre(route["choix"]["ordre"]), _txt_ordre(au_["ordre"]), _val_lat(choix_total_tmp(route)), _val_lat(au_["total"]))))
                    statut_int = False
            except Exception:
                pass
        if statut_int is True:
            _P(_T("Ainsi $\\int_D|f|\\,d\\lambda<+\\infty$ : $f\\in L^{1}(D)$. Le théorème de Fubini s'applique : $I$ se calcule par intégrales itérées, dans n'importe quel ordre."))
        elif statut_int is False:
            _P(_T("Ainsi $\\int_D|f|\\,d\\lambda=+\\infty$ : $f\\notin L^{1}(D)$. Le théorème de Fubini ne s'applique pas, et $\\int_D f\\,d\\lambda$ n'est pas définie au sens de Lebesgue."))
        else:
            _P(_T("L'intégrabilité de $f$ n'a pas pu être établie par les arguments précédents : le calcul itéré ci-dessous n'est justifié que si $f\\in L^{1}(D)$."))

    # ---------------- Symétrie ----------------
    zero_par_symetrie = False
    if signe == "mixte" and statut_int is True:
        for v in V:
            try:
                if sp.simplify(f.subs(v, -v) + f) == 0 and _symetrique(pieces, V, v):
                    _P(_T("$D$ est symétrique par rapport à l'hyperplan $%s=0$ et $f(\\ldots,-%s,\\ldots)=-f(\\ldots,%s,\\ldots)$ : $f$ est impaire en $%s$. Comme $f$ est intégrable, le changement de variable $%s\\mapsto-%s$ (qui conserve la mesure) donne $I=-I$, donc $I=0$." % (sp.latex(v), sp.latex(v), sp.latex(v), sp.latex(v), sp.latex(v), sp.latex(v))))
                    zero_par_symetrie = True
                    break
            except Exception:
                pass

    # ---------------- Changement de variables ----------------
    if nom_route == "changement":
        _titre("Changement de variables")
        if chg["kind"] in ("polaire", "cylindrique", "spherique", "sphérique"):
            nom, ls = _decrit_changement(chg, pieces, V, k)
            _P(_T("La géométrie du domaine invite aux coordonnées %s : $%s$." % (nom, ls)))
            if chg.get("generalise"):
                _P(_T("Ce sont des coordonnées %s généralisées : composition d'une translation et de dilatations d'axes (difféomorphismes affines) avec les coordonnées %s usuelles, toutes de classe $C^{1}$ avec une réciproque de classe $C^{1}$." % (nom, nom)))
            _P(_T("Cette application est un $C^{1}$-difféomorphisme de l'ouvert correspondant sur $D$ privé d'un ensemble négligeable, de jacobien $|J|=%s$." % sp.latex(chg["jac"])))
            if len(choix["cells"]) == 1 and len(chg["pieces"]) == 1:
                _P(_T("Le nouveau domaine est $\\Delta=" + _lat_ens_cell(chg["V"], choix["cells"][0][1]) + "$ et l'intégrande devient $(f\\circ\\Phi)|J|=%s$." % sp.latex(choix["cells"][0][0])))
            else:
                _P(_T("Le nouveau domaine $\\Delta$ est réunion de %d morceaux disjoints (à un ensemble négligeable près) :" % len(choix["cells"])))
                for i, (fi, c) in enumerate(choix["cells"]):
                    _P("\\Delta_{%d}=" % (i + 1) + _lat_ens_cell(chg["V"], c))
            _P(_T("Le théorème de changement de variables s'applique à $f\\ge0$ mesurable (valeur dans $[0,+\\infty]$) comme à $f\\in L^{1}(D)$ ($|f|$ se transforme de la même façon, ce qui conserve l'intégrabilité) : $\\int_D f\\,d\\lambda=\\int_{\\Delta}(f\\circ\\Phi)\\,|J|\\,d\\lambda$."))
        else:
            _P(_T("Le domaine et la fonction invitent à un changement de variables %s." % chg["nom"]))
            for l_ in _lignes_hypotheses(chg, V, pieces, None):
                _P(l_)
            cond_d = []
            if len(choix["cells"]) == 1 and len(chg["pieces"]) == 1:
                _P(_T("Le nouveau domaine est $\\Delta=" + _lat_ens_cell(chg["V"], choix["cells"][0][1]) + "$ et l'intégrande devient $(f\\circ\\Phi)|J|=%s$." % sp.latex(choix["cells"][0][0])))
            else:
                _P(_T("Le nouveau domaine $\\Delta$ est réunion de %d morceaux disjoints (à un ensemble négligeable près) :" % len(choix["cells"])))
                for i, (fi, c) in enumerate(choix["cells"]):
                    _P("\\Delta_{%d}=" % (i + 1) + _lat_ens_cell(chg["V"], c))
            _P(_T("Le théorème de changement de variables s'applique à $f\\ge0$ mesurable (valeur dans $[0,+\\infty]$) comme à $f\\in L^{1}(D)$ ($|f|$ se transforme de la même façon, ce qui conserve l'intégrabilité) : $\\int_D f\\,d\\lambda=\\int_{\\Delta}(f\\circ\\Phi)\\,|J|\\,d\\lambda$."))
        Vcalc = chg["V"]
    else:
        Vcalc = V

    # ---------------- Calcul ----------------
    _titre("Calcul")
    ph = _phrase_ordre(cart, choix) if nom_route != "changement" else None
    if ph:
        _P(ph)
    ord_txt = _txt_ordre(choix["ordre"])
    if statut_int is False:
        _P(_T("Puisque $f\\notin L^{1}(D)$, Fubini ne garantit rien ; on calcule seulement, à titre d'illustration, l'intégrale itérée dans l'ordre $%s$." % ord_txt))
    elif len(Vcalc) > 1:
        _P(_T("Par Fubini–Tonelli, on intègre dans l'ordre $%s$ (la première variable est la plus extérieure)." % ord_txt))
    if zero_par_symetrie:
        _P(_T("Le calcul itéré confirme ce résultat :"))
    _rediger_calcul(choix["resultats"], V, Vcalc, f, nom_route, chg, None, unique_piece=(chg is None or len(chg["pieces"]) == 1))

    # ---------------- Bilan ----------------
    _titre("Bilan")
    total = choix["total"]
    if statut_int is False:
        _P(_T("$f$ n'est pas intégrable sur $D$ : le calcul itéré dans l'ordre $%s$ donne $%s$, mais cette valeur n'a pas de sens pour l'intégrale de Lebesgue." % (ord_txt, _val_lat(total))))
        au = route.get("autre")
        if au is not None:
            try:
                diff = (au["total"] != total) and not (au["total"] == total)
                if diff and not sp.simplify(au["total"] - total) == 0 if au["total"] not in (sp.oo, -sp.oo) and total not in (sp.oo, -sp.oo) else au["total"] != total:
                    _P(_T("Dans l'ordre $%s$ on trouve $%s$ : les deux intégrales itérées existent mais diffèrent. C'est précisément ce que Fubini ne garantit plus sans l'hypothèse $f\\in L^{1}(D)$ : c'est un contre-exemple à l'interversion des intégrales." % (_txt_ordre(au["ordre"]), _val_lat(au["total"]))))
            except Exception:
                pass
    else:
        if total == sp.oo:
            _P(_T("$I=+\\infty$ : l'intégrale de $f\\ge0$ diverge (Tonelli)."))
        else:
            _P("I=" + _val_lat(total))
    # autre ordre
    autre = route.get("autre")
    if autre is not None and statut_int is not False:
        try:
            if autre["total"] == total and total in (sp.oo, -sp.oo):
                same = True
            elif autre["total"] in (sp.oo, -sp.oo) or total in (sp.oo, -sp.oo):
                same = False
            else:
                same = sp.simplify(autre["total"] - total) == 0
        except Exception:
            same = None
        if same:
            _P(_T("Contrôle : l'ordre $%s$ conduit à la même valeur $%s$, comme l'affirme Fubini–Tonelli." % (_txt_ordre(autre["ordre"]), _val_lat(total))))
        elif same is False:
            _P(_T("Attention : l'ordre $%s$ conduit à $%s$, valeur différente : l'une des intégrales itérées est à revoir (ou $f$ n'est pas intégrable)." % (_txt_ordre(autre["ordre"]), _val_lat(autre["total"]))))
    # contrôle numérique
    if total not in (sp.oo, -sp.oo) and statut_int is not False:
        cn = _controle_numerique(f_route_cells, Vcalc)
        if cn is not None:
            try:
                ex = float(sp.N(total))
                ecart = abs(ex - cn)
                if ecart < 1e-4 * (1 + abs(ex)):
                    _P(_T("Contrôle numérique indicatif : valeur approchée $%.6g$, en accord avec la valeur exacte." % cn))
                else:
                    _P(_T("Contrôle numérique indicatif : valeur approchée $%.6g$, qui s'écarte de la valeur exacte trouvée ($%.6g$)." % (cn, ex)))
            except Exception:
                pass

    # ---------------- Graphes ----------------
    _titre("Graphes")
    try:
        _graphes(f, pieces, V, (cart["choix"]["cells"] if cart["ok"] else (cart.get("cells_any") or [])), None)
    except Exception as e:  # pragma: no cover
        _rouge("Les graphes n'ont pas pu être produits.")


def etude_integrale_multiple(fstr, dstr, vstr, loc, N=None):
    _ALERTE.clear()
    try:
        _etude_interne(fstr, dstr, vstr, loc, N)
    except _NonSupporte as e:
        _rouge(str(e))
    except Exception as e:
        _rouge("Le calcul n'a pas pu être mené (%s)." % str(e)[:100])
    if _ALERTE:
        _rouge("Un calcul intermédiaire a produit une valeur indéfinie ; il a été écarté et le résultat est à prendre avec précaution.")


# -------------------------------------------------------------
# Rédaction du calcul : chaîne d'égalités alignées (de haut en bas)
# -------------------------------------------------------------
_GAUSS_FAIT = []


def _lignes_gauss(g, v, lo, hi, val):
    """Justification d'une intégrale de type Gauss (bornes infinies) ; [] si la forme n'est pas reconnue."""
    fg = _forme_gauss(g, v)
    if fg is None or _gauss_valeur(g, v, lo, hi) is None:
        return []
    C, P, a, b = fg
    sv = sp.latex(v)
    lg = []
    if "gauss" not in _GAUSS_FAIT:
        _GAUSS_FAIT.append("gauss")
        lg.append(_T(r"Intégrale de Gauss : $G=\int_{-\infty}^{+\infty}e^{-t^{2}}\,dt=\sqrt{\pi}$. En effet $G>0$ et, par Tonelli (la fonction est positive), $G^{2}=\iint_{\mathbb{R}^{2}}e^{-(s^{2}+t^{2})}\,ds\,dt$ ; en coordonnées polaires, $G^{2}=\int_{0}^{2\pi}\!\int_{0}^{+\infty}re^{-r^{2}}\,dr\,d\theta=2\pi\left[-\tfrac12e^{-r^{2}}\right]_{0}^{+\infty}=\pi$."))
    sa = sp.latex(a)
    un = (a == 1)
    ac = "" if un else sa
    rac = r"\sqrt{\pi}" if un else r"\sqrt{\pi/%s}" % sa
    if lo == -sp.oo:
        c = sp.simplify(b / (2 * a))
        wv = v
        if b != 0:
            wv = sp.Symbol("w", real=True)
            can = r"-%s%s^{2}+%s%s=-%s\left(%s-%s\right)^{2}+%s" % (ac, sv, ("" if b == 1 else sp.latex(b)), sv, ac, sv, sp.latex(c), sp.latex(sp.simplify(b ** 2 / (4 * a))))
            lg.append(_T("Forme canonique : $" + can + "$."))
            Pw = sp.expand(P.subs(v, wv + c))
            lg.append(_T(r"La translation $w=%s-%s$ conserve la mesure de Lebesgue : $\int_{\mathbb{R}}%s\,e^{-%s(%s-%s)^{2}}\,d%s=\int_{\mathbb{R}}\left(%s\right)e^{-%sw^{2}}\,dw$." % (sv, sp.latex(c), "\\left(" + sp.latex(P) + "\\right)" if sp.Poly(P, v).degree() > 0 else sp.latex(P), ac, sv, sp.latex(c), sv, sp.latex(Pw), ac)))
        deg = sp.Poly(P, v).degree()
        if deg > 0:
            lg.append(_T(r"Moments : pour $a>0$, $\int_{\mathbb{R}}w^{2j+1}e^{-aw^{2}}dw=0$ (intégrande impaire, intégrable) et, par intégration par parties, $\int_{\mathbb{R}}w^{2j}e^{-aw^{2}}dw=\frac{2j-1}{2a}\int_{\mathbb{R}}w^{2j-2}e^{-aw^{2}}dw$, avec $\int_{\mathbb{R}}e^{-aw^{2}}dw=\sqrt{\pi/a}$ (changement $t=\sqrt{a}\,w$ dans $G$)." + (" Ici $a=%s$." % sa if not un else "")))
        elif not un:
            lg.append(_T(r"Le changement de variable linéaire $t=\sqrt{%s}\,w$ donne $\int_{\mathbb{R}}e^{-%sw^{2}}\,dw=\frac{G}{\sqrt{%s}}=\sqrt{\pi/%s}$." % (sa, sa, sa, sa)))
        else:
            lg.append(_T(r"Ici $\int_{\mathbb{R}}e^{-w^{2}}\,dw=G=\sqrt{\pi}$."))
    else:
        expo = [j for (j,), _c in sp.Poly(P, v).terms()]
        if any(j % 2 == 0 for j in expo):
            lg.append(_T(r"Pour une puissance paire, l'intégrande est paire : $\int_{0}^{+\infty}%s^{j}e^{-%s%s^{2}}d%s=\frac12\int_{\mathbb{R}}%s^{j}e^{-%s%s^{2}}d%s$, et l'on utilise les moments de Gauss (par intégration par parties, $\int_{\mathbb{R}}t^{2j}e^{-at^{2}}dt=\frac{2j-1}{2a}\int_{\mathbb{R}}t^{2j-2}e^{-at^{2}}dt$, avec $\int_{\mathbb{R}}e^{-at^{2}}dt=\sqrt{\pi/a}$)." % (sv, ac, sv, sv, sv, ac, sv, sv)))
        if any(j % 2 == 1 for j in expo):
            lg.append(_T(r"Pour une puissance impaire, la primitive est élémentaire : $\int %s^{2i+1}e^{-a%s^{2}}d%s$ se calcule par le changement $u=%s^{2}$, ce qui donne $\int_{0}^{+\infty}%s^{2i+1}e^{-a%s^{2}}d%s=\frac{i!}{2a^{i+1}}$." % (sv, sv, sv, sv, sv, sv, sv)))
        if not un:
            lg.append(_T("Ici $a=%s$." % sa))
    lg.append(_T(r"Ainsi $\int_{%s}^{+\infty}%s\,d%s=%s$." % ("-\\infty" if lo == -sp.oo else "0", sp.latex(g), sv, sp.latex(val))))
    return lg


def _lp(e):
    """LaTeX de e, entre parenthèses si c'est une somme (intégrande)."""
    if e == sp.oo:
        return r"+\infty"
    s = sp.latex(e)
    if isinstance(e, sp.Basic) and e.is_Add:
        return r"\left(" + s + r"\right)"
    return s


def _imbrique(cell, expr, upto):
    """\\int_{lo1}^{hi1}...\\int_{lo_upto}^{hi_upto} expr dv_upto ... dv_1"""
    if upto <= 0:
        return expr
    pref = "".join(r"\int_{" + _lb(lo) + "}^{" + _lb(hi) + r"}\!" for v, lo, hi in cell[:upto])
    diff = "".join(r"\,d" + sp.latex(v) for v, _, _ in reversed(cell[:upto]))
    return pref + expr + diff


def _signe_int(k):
    if k == 1:
        return r"\int"
    if k == 2:
        return r"\iint"
    if k == 3:
        return r"\iiint"
    return r"\int\!\cdots\!\int"


_ROMAN = "abcdefghijklmnopqrstuvwxyz"


def _bloc_aligne(lignes):
    """lignes : liste de (debut, expr, commentaire). Renvoie un bloc LaTeX aligné."""
    corps = []
    for deb, expr, com in lignes:
        s = deb + " &= " + expr
        if com:
            s += r" && " + _T("(" + com + ")")
        corps.append(s)
    return r"\begin{aligned} " + r" \\ ".join(corps) + r" \end{aligned}"


def _chaine_cellule(f_, cell, val, steps, nom, details, gauche_debut, ordre_gauche=None, com0="Fubini–Tonelli"):
    """Lignes (debut, expr, commentaire) de l'intégrale itérée d'une cellule, du haut vers le bas."""
    k = len(cell)
    L = []
    L.append((gauche_debut, _imbrique(cell, _lp(f_), k), com0 if k > 1 else ""))
    for s_idx, stp in enumerate(steps):
        v, lo, hi, g, F, vl = stp["v"], stp["lo"], stp["hi"], stp["g"], stp["F"], stp["val"]
        i = k - 1 - s_idx
        sv = sp.latex(v)
        com = ""
        if F is not None:
            lg = []
            bloc = None
            try:
                pbu, pbv = lo in (sp.oo, -sp.oo), hi in (sp.oo, -sp.oo)
                if not (pbu and pbv) and vl not in (sp.oo, -sp.oo):
                    bloc = _L._chaine_int(g, v, lo, hi, pbu, pbv, vl)
            except Exception:
                bloc = None
            if bloc:
                lg = [bloc]
                non_triv = bloc.count("\\\\") >= 3
            else:
                try:
                    lg = _L._narrer_primitive(g, v, F) or []
                except Exception:
                    lg = []
                non_triv = any(("On pose" in l_ or "parties" in l_ or "éléments simples" in l_ or "réécrit" in l_) for l_ in lg)
            if non_triv:
                lettre = _ROMAN[len(details) % 26]
                details.append(("primitive de $%s$ en $%s$" % (sp.latex(g), sv), lg))
                com = "primitive en $%s$, détail %s" % (sv, lettre)
            else:
                com = "primitive en $%s$" % sv
            br = r"\left[" + sp.latex(F) + r"\right]_{" + _lb(lo) + "}^{" + _lb(hi) + "}"
            L.append(("", _imbrique(cell, br, i), com))
            L.append(("", _imbrique(cell, _lp(vl), i), ""))
        else:
            lg = _lignes_gauss(g, v, lo, hi, vl)
            if lg:
                lettre = _ROMAN[len(details) % 26]
                details.append(("intégrale de type Gauss en $%s$" % sv, lg))
                com = "intégrale de Gauss, détail %s" % lettre
            else:
                com = "calcul direct en $%s$" % sv
            L.append(("", _imbrique(cell, _lp(vl), i), com))
        if vl in (sp.oo, -sp.oo):
            break
    return L


def _somme_lat(vals):
    if any(v == sp.oo for v in vals):
        return r"+\infty"
    try:
        return sp.latex(sp.Add(*vals, evaluate=False))
    except Exception:
        return " + ".join(_val_lat(v) for v in vals)


def _rediger_calcul(resultats, V, Vcalc, f, nom_route, chg, aff_dom, unique_piece=True):
    """Écrit la chaîne d'égalités du calcul. Renvoie (total_ecrit)."""
    _GAUSS_FAIT.clear()
    k = len(V)
    kc = len(Vcalc)
    multiple = len(resultats) > 1
    details = []
    blocs = []
    # en-tête commun
    tete = []
    intro_f = sp.latex(f)
    if nom_route == "changement":
        sg = _signe_int(k)
        tete.append(("I", sg + r"_{D} " + _lp(f) + r"\," + _dx(V), ""))
        fn = resultats[0][0] if not multiple else None
        nm = ",".join(sp.latex(v) for v in Vcalc)
        if len(resultats) == 1 and unique_piece:
            tete.append(("", sg + r"_{\Delta} " + _lp(resultats[0][0]) + r"\," + _dx(Vcalc), "changement de variables"))
        else:
            tete.append(("", sg + r"_{\Delta} f\left(\Phi(" + nm + r")\right)\,|J|\," + _dx(Vcalc), "changement de variables"))
    else:
        tete.append(("I", _signe_int(k) + r"_{D} " + _lp(f) + r"\," + _dx(V), ""))
    if not multiple:
        f_, c, val, steps = resultats[0]
        L = tete + _chaine_cellule(f_, c, val, steps, "I", details, "", com0="Fubini–Tonelli")[0:]
        # première ligne de la chaîne sans début
        _P(_bloc_aligne(L))
    else:
        somme = " + ".join("I_{%d}" % (i + 1) for i in range(len(resultats)))
        L = tete + [("", somme, "$D$ réunion des $D_i$ disjoints à un négligeable près")]
        _P(_bloc_aligne(L))
        vals = []
        for i, (f_, c, val, steps) in enumerate(resultats):
            Li = _chaine_cellule(f_, c, val, steps, "I_{%d}" % (i + 1), details, "I_{%d}" % (i + 1))
            _P(_bloc_aligne(Li))
            vals.append(val)
        S = [("I", _somme_lat(vals), "")]
        S.append(("", _val_lat(sum(vals, sp.Integer(0)) if sp.oo not in vals else sp.oo), ""))
        _P(_bloc_aligne(S))
    if details:
        _titre("Détails des calculs de primitives")
        for j, (titre, lg) in enumerate(details):
            _P(_T("(%s) %s :" % (_ROMAN[j % 26], titre)))
            for l_ in lg:
                _P(l_)


# -------------------------------------------------------------
# Changements de variables généraux : linéaires et hyperboliques (u = xy, v = y/x)
# -------------------------------------------------------------
def _norm_vec(vec):
    den = 1
    for c in vec:
        den = sp.ilcm(den, sp.Rational(c).q)
    ent = [int(sp.Rational(c) * den) for c in vec]
    g = 0
    for c in ent:
        g = sp.igcd(g, abs(c))
    g = g or 1
    ent = [c // g for c in ent]
    for c in ent:
        if c != 0:
            if c < 0:
                ent = [-t for t in ent]
            break
    return tuple(ent)


def _vecteur_lineaire(e, V):
    """Coefficients (rationnels) de e s'il est affine en V avec au moins 2 variables, sinon None."""
    try:
        P = sp.Poly(sp.expand(e), *V)
    except Exception:
        return None
    if P.total_degree() != 1:
        return None
    d = P.as_dict()
    k = len(V)
    vec = [d.get(tuple(1 if j == i else 0 for j in range(k)), sp.Integer(0)) for i in range(k)]
    if not all(sp.Rational(c).is_Rational for c in vec):
        return None
    if sum(1 for c in vec if c != 0) < 2:
        return None
    return _norm_vec(vec)


def _pool_formes(f, pieces, V):
    pool = []
    def add(vec):
        if vec is not None and vec not in pool:
            pool.append(vec)
    for pc in pieces:
        for g, _s in pc["cons"]:
            add(_vecteur_lineaire(g, V))
        for e in sp.preorder_traversal(pc["f"]):
            if e.is_Add and sum(1 for v in V if e.has(v)) >= 2:
                add(_vecteur_lineaire(e, V))
    return pool[:6]


def _matrices_candidates(pool, k, maximum=8):
    res = []
    axes = [tuple(1 if j == i else 0 for j in range(k)) for i in range(k)]
    for m in range(min(k, len(pool)), 0, -1):
        for sub in itertools.combinations(pool, m):
            if sp.Matrix(list(sub)).rank() < m:
                continue
            for comp in itertools.combinations(axes, k - m):
                A = sp.Matrix(list(sub) + list(comp))
                if A.det() != 0:
                    res.append(A)
                    break
        if len(res) >= maximum:
            break
    return res[:maximum]


def _transforme_pieces(pieces, sub, jac):
    """Applique x = sub(u) ; retourne les nouvelles pièces (contraintes triviales traitées)."""
    nouvelles = []
    for pc in pieces:
        cons_n = []
        vide = False
        for g, s in pc["cons"]:
            e = sp.simplify(sp.expand(g.subs(sub, simultaneous=True)))
            if e.is_number:
                val = float(e)
                if (val < 0) if s else (val <= 0):
                    continue
                vide = True
                break
            cons_n.append((e, s))
        if vide:
            continue
        fn = sp.simplify(pc["f"].subs(sub, simultaneous=True) * jac)
        nouvelles.append(dict(f=fn, cons=cons_n))
    return nouvelles


def _couplees(pieces, Vn):
    n = 0
    for pc in pieces:
        for g, _s in pc["cons"]:
            if sum(1 for v in Vn if g.has(v)) >= 2:
                n += 1
    return n


def _noms_u(V, k):
    base = ["u", "v", "w"] if k <= 3 else ["u%d" % (i + 1) for i in range(k)]
    return _noms_libres(V, base[:k])


def _candidats_lineaires(f, pieces, V):
    k = len(V)
    pool = _pool_formes(f, pieces, V)
    if not pool:
        return []
    res = []
    for A in _matrices_candidates(pool, k):
        U = _noms_u(V, k)
        B = A.inv()
        sub = {V[i]: sum((B[i, j] * U[j] for j in range(k)), sp.Integer(0)) for i in range(k)}
        jac = sp.Abs(B.det())
        try:
            pn = _transforme_pieces(pieces, sub, jac)
        except Exception:
            continue
        if not pn:
            continue
        cpl = _couplees(pn, U)
        cplf = sum(sp.count_ops(pc["f"]) for pc in pn)
        res.append(dict(kind="lineaire", nom="linéaire", A=A, B=B, sub=sub, jac=sp.nsimplify(jac), V=list(U), pieces=pn,
                        pre=4 * cpl + 0.05 * cplf, couplees=cpl, grp=None, angles=[], hyp=None))
    res.sort(key=lambda c: c["pre"])
    return res[:3]


def _lp_minimum(cons, V, v):
    """Minimum de v sur le polyèdre défini par les contraintes affines de cons (None si non borné/échec)."""
    try:
        from scipy.optimize import linprog
        k = len(V)
        A, b = [], []
        for g, _s in cons:
            P = sp.Poly(sp.expand(g), *V)
            if P.total_degree() > 1:
                continue
            d = P.as_dict()
            A.append([float(d.get(tuple(1 if j == i else 0 for j in range(k)), 0)) for i in range(k)])
            b.append(-float(d.get(tuple(0 for _ in range(k)), 0)))
        if not A:
            return None
        c = [1.0 if w == v else 0.0 for w in V]
        r = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * k, method="highs")
        if r.status == 0:
            return float(r.fun)
    except Exception:
        return None
    return None


def _positif_sur_D(pieces, V, v):
    """Vrai si D est contenu dans {v > 0} à un ensemble négligeable près."""
    try:
        if all(v in _variables_positives(pc["cons"], V) for pc in pieces):
            return True
        if all(_lp_minimum(pc["cons"], V, v) is not None and _lp_minimum(pc["cons"], V, v) >= -1e-9 for pc in pieces):
            return True
        autres = [dict(f=pc["f"], cons=list(pc["cons"]) + [(v, False)]) for pc in pieces]
        res, ech = _ordres_possibles(autres, V, budget=6.0, maximum=1)
        return (not res) and ech is None
    except Exception:
        return False


def _contrainte_hyperbolique(g, x, y, u, v):
    """Réécrit g <= 0 (x,y>0) en une contrainte sur v seul (rapport y/x) ou u seul (produit xy)."""
    try:
        g1 = sp.factor(sp.simplify(g.subs(y, v * x)))
        q, xm = g1.as_independent(x, as_Add=False)
        if not q.has(x) and (xm == 1 or xm == x or (xm.is_Pow and xm.base == x)):
            return sp.expand(q), v
        g2 = sp.factor(sp.simplify(g.subs(y, u / x)))
        q, xm = g2.as_independent(x, as_Add=False)
        if not q.has(x) and (xm == 1 or xm == x or (xm.is_Pow and xm.base == x)):
            return sp.expand(q), u
    except Exception:
        pass
    return None


def _candidats_hyperboliques(f, pieces, V):
    cands = []
    k = len(V)
    if k < 2:
        return cands
    for x, y in itertools.permutations(V, 2):
        if V.index(x) > V.index(y):
            continue
        if not (_positif_sur_D(pieces, V, x) and _positif_sur_D(pieces, V, y)):
            continue
        U = _noms_libres(V, ["u", "v"])
        u, v = U
        up, vp = sp.Symbol(u.name, positive=True), sp.Symbol(v.name, positive=True)
        ok = True
        nouvelles = []
        for pc in pieces:
            cons_n = []
            for g, s in pc["cons"]:
                if not (g.has(x) or g.has(y)):
                    cons_n.append((g, s))
                    continue
                r = _contrainte_hyperbolique(g, x, y, u, v)
                if r is None:
                    ok = False
                    break
                q, _quoi = r
                if q.is_number:
                    val = float(q)
                    if (val < 0) if s else (val <= 0):
                        continue
                    ok = False
                    break
                cons_n.append((q, s))
            if not ok:
                break
            cons_n += [(-u, True), (-v, True)]
            sub_p = {x: sp.sqrt(up / vp), y: sp.sqrt(up * vp)}
            fn = sp.simplify(pc["f"].subs(sub_p, simultaneous=True) * sp.Rational(1, 2) / vp)
            fn = fn.subs({up: u, vp: v})
            nouvelles.append(dict(f=fn, cons=cons_n))
        if not ok or not nouvelles:
            continue
        cpl = _couplees(nouvelles, [u, v] + [w for w in V if w not in (x, y)])
        cplf = sum(sp.count_ops(pc["f"]) for pc in nouvelles)
        sub = {x: sp.sqrt(u / v), y: sp.sqrt(u * v)}
        Vn = [u, v] + [w for w in V if w not in (x, y)]
        cands.append(dict(kind="hyperbolique", nom="hyperboliques", xy=(x, y), sub=sub, jac=sp.Rational(1, 2) / v, V=Vn, pieces=nouvelles,
                          pre=4 * cpl + 0.05 * cplf, couplees=cpl, grp=None, angles=[], hyp=None))
    cands.sort(key=lambda c: c["pre"])
    return cands[:2]


def _lignes_hypotheses(c, V, pieces_orig, D_desc):
    """Vérification rédigée des hypothèses du théorème de changement de variables (cas non classique)."""
    k = len(V)
    Un = c["V"]
    lg = []
    nm = lambda L: "(" + ",".join(sp.latex(t) for t in L) + ")"
    if c["kind"] == "lineaire":
        A, B = c["A"], c["B"]
        ls = [sp.latex(Un[i]) + "=" + sp.latex(sum((A[i, j] * V[j] for j in range(k)), sp.Integer(0))) for i in range(k)]
        lg.append(_T("On pose $" + r",\quad ".join(ls) + "$, soit $" + nm(Un) + "=A" + nm(V) + "$ avec $A=" + sp.latex(A) + "$."))
        lg.append(_T("$\\det A=" + sp.latex(A.det()) + "\\neq0$ : $A$ est inversible, donc $A:\\mathbb{R}^{%d}\\to\\mathbb{R}^{%d}$ est un $C^{1}$-difféomorphisme de $\\mathbb{R}^{%d}$ sur lui-même, de réciproque $\\Phi=A^{-1}$ : $" % (k, k, k) + r",\quad ".join(sp.latex(V[i]) + "=" + sp.latex(c["sub"][V[i]]) for i in range(k)) + "$."))
        lg.append(_T("Son jacobien est constant : $|J|=|\\det A^{-1}|=" + sp.latex(c["jac"]) + "\\neq0$."))
    else:
        x, y = c["xy"]
        u, v = Un[0], Un[1]
        lg.append(_T("On pose $%s=%s\\,%s$ et $%s=\\dfrac{%s}{%s}$, ce qui suppose $%s>0$ et $%s>0$ ; or $D$ est contenu dans $]0,+\\infty[^{2}$ pour ces deux variables (à un ensemble négligeable près, les bords étant de mesure nulle)." % (sp.latex(u), sp.latex(x), sp.latex(y), sp.latex(v), sp.latex(y), sp.latex(x), sp.latex(x), sp.latex(y))))
        lg.append(_T("L'application $\\Psi:(%s,%s)\\mapsto(%s%s,\\ %s/%s)$ est de classe $C^{1}$ de $]0,+\\infty[^{2}$ dans lui-même, bijective, de réciproque $\\Phi:(%s,%s)\\mapsto\\left(\\sqrt{%s/%s},\\ \\sqrt{%s%s}\\right)$ elle aussi de classe $C^{1}$ : c'est un $C^{1}$-difféomorphisme de $]0,+\\infty[^{2}$." % (sp.latex(x), sp.latex(y), sp.latex(x), sp.latex(y), sp.latex(y), sp.latex(x), sp.latex(u), sp.latex(v), sp.latex(u), sp.latex(v), sp.latex(u), sp.latex(v))))
        lg.append(_T("Jacobien de $\\Phi$ : $|J|=\\left|\\det\\frac{\\partial(%s,%s)}{\\partial(%s,%s)}\\right|=%s\\neq0$ sur $]0,+\\infty[^{2}$." % (sp.latex(x), sp.latex(y), sp.latex(u), sp.latex(v), sp.latex(c["jac"]))))
    return lg


def _cout_route(res):
    """Coût approximatif d'une route résolue (plus petit = plus simple)."""
    if not res or not res.get("ok"):
        return 1e9
    ch = res["choix"]
    cout = 6.0 * (len(ch["resultats"]) - 1)
    for f_, c, val, steps in ch["resultats"]:
        for stp in steps:
            cout += 1 + 0.12 * sp.count_ops(stp["g"])
            for b in (stp["lo"], stp["hi"]):
                if b not in (sp.oo, -sp.oo):
                    if not b.is_number:
                        cout += 1.2
                    if _a_racine(b):
                        cout += 3
    return cout
