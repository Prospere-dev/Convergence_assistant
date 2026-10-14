# ============================================================
# app.py (coeur SymPy)
# Fichier destiné à être importé par Flask (routes séparées).
# Toutes les sorties passent par st.latex()/st.info()/st.error().
# ============================================================

from flask import Flask, request, jsonify, render_template
from sympy import S
import sympy as sp
from sympy import (
    sin, cos, tan,
    asin, acos, atan,
    sinh, cosh, tanh,
    asinh, acosh, atanh,
    exp, log, ln, sqrt,
    pi, E, Abs, sign,
    erf, gamma, zeta,
    floor, ceiling
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sympy.calculus.util import continuous_domain
from sympy import S
from sympy.assumptions import Q, ask
from sympy.core.relational import Relational, Equality
import io
import base64

# ------------------------------------------------------------
# Symboles globaux
# ------------------------------------------------------------
k = sp.symbols("k", integer=True, positive=True)


# ============================================================
# Faux Streamlit : st.latex(), st.info(), st.error(), st.pyplot()
# ============================================================
class _StreamlitCompat:
    def __init__(self):
        self._out = []

    def set_out(self, out):
        self._out = out

    def latex(self, msg):
        self._out.append({"type": "latex", "content": msg})

    def info(self, msg):
        self._out.append({"type": "info", "content": str(msg)})

    def error(self, msg):
        self._out.append({"type": "error", "content": str(msg)})

    def pyplot(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        self._out.append({"type": "plot", "content": b64})


st = _StreamlitCompat()




def facteur(expression):# 
    #retourne le facteur multiplicatif rationnel de l'expression et retourne 1 s'il ny'en a pas de visible ou si le facteur est irrationnel
    # Pour les rationnels
    facteur , reste =  expression.as_coeff_Mul()# Je sépare le coefficient rationnel et le reste de   l'expression (qui est une fonction de n)
    return facteur

def facteur2(expression) : 
    expression_approx = expression.evalf()# On evalue expression en fonction de n, toutes les constantes irrationnelles y sont évaluées( n lui n'est pas evalué)
    facteur_approx = facteur(expression_approx)
    quotient = sp.simplify(expression_approx/expression)# Ce quotient est purement réel mathématiquement (expression.approx est au numérateur pour eviter les inversions des approximations) 
    # On remarquera que quotient = facteur_approx/facteur aussi donc
    facteur = sp.simplify(facteur_approx / quotient)  # on obtient 1.0*facteur
    return facteur 





# ============================================================
# Fonctions bornées connues
# ============================================================
BORNES_CONNUES = {
    sp.sin: (-1, 1),
    sp.cos: (-1, 1),
    sp.sign: (-1, 1),
    sp.Abs: (0, None),
}


# ============================================================
# Outils de signe / encadrements
# ============================================================
def signe_asymptotique(expr, n):
    # 1) direct
    if expr.is_positive:
        return 1
    if expr.is_negative:
        return -1

    # 2) limite
    try:
        L = sp.limit(expr, n, sp.oo)
        if L.is_positive:
            return 1
        if L.is_negative:
            return -1
    except:
        pass

    # 3) terme dominant
    try:
        lead = sp.simplify(expr.as_leading_term(n))
        if lead.is_positive:
            return 1
        if lead.is_negative:
            return -1
    except:
        pass

    return 0


def min_borne(a, b):
    d = sp.simplify(a - b)
    if d.is_nonpositive:
        return a
    if d.is_nonnegative:
        return b
    return a


def max_borne(a, b):
    d = sp.simplify(a - b)
    if d.is_nonnegative:
        return a
    if d.is_nonpositive:
        return b
    return b


def borne_constante_variable(expr, n):
    if expr.is_Number:
        return expr, expr
    if expr == n:
        return expr, expr
    # (-1)**(quelquechose en n) : borné dans [-1,1]
    if expr.is_Pow and expr.base == -1 and expr.exp.has(n):
        return sp.Integer(-1), sp.Integer(1)
    return None


def borne_fonction(expr):
    if expr.is_Function and expr.func in BORNES_CONNUES:
        bmin, bmax = BORNES_CONNUES[expr.func]
        if bmin is not None and bmax is not None:
            return sp.sympify(bmin), sp.sympify(bmax)
        return expr, expr
    return None


def encadrements(expr, n):
    # constantes / n / (-1)^n
    r = borne_constante_variable(expr, n)
    if r is not None:
        return r

    # sin/cos/...
    r = borne_fonction(expr)
    if r is not None:
        return r

    # somme
    if expr.is_Add:
        mi, ma = 0, 0
        for t in expr.args:
            tmi, tma = encadrements(t, n)
            mi += tmi
            ma += tma
        return sp.simplify(mi), sp.simplify(ma)

    # produit
    if expr.is_Mul:
        mi, ma = 1, 1
        for f in expr.args:
            fmi, fma = encadrements(f, n)
            c1 = min_borne(mi * fmi, mi * fma)
            c2 = min_borne(ma * fmi, ma * fma)
            mi = min_borne(c1, c2)
            c3 = max_borne(mi * fmi, mi * fma)
            c4 = max_borne(ma * fmi, ma * fma)
            ma = max_borne(c3, c4)
        return sp.simplify(mi), sp.simplify(ma)

    # puissance : uniquement cas sûrs (entiers, -1)
    if expr.is_Pow:
        base, exp = expr.as_base_exp()
        bmin, bmax = encadrements(base, n)

        if exp == -1:
            # inversion seulement si signe contrôlé
            if bmin.is_positive:
                return sp.simplify(1 / bmax), sp.simplify(1 / bmin)
            if bmax.is_negative:
                return sp.simplify(1 / bmin), sp.simplify(1 / bmax)
            return expr, expr

        if exp.is_Integer and exp % 2 == 0:
            if bmin.is_nonnegative:
                return sp.simplify(bmin**exp), sp.simplify(bmax**exp)
            if bmax.is_nonpositive:
                return sp.simplify(bmax**exp), sp.simplify(bmin**exp)
            return sp.Integer(0), sp.simplify(max(abs(bmin), abs(bmax)) ** exp)

        if exp.is_Integer and exp % 2 == 1:
            return sp.simplify(bmin**exp), sp.simplify(bmax**exp)

        return expr, expr

    # fraction : seulement si signe dénominateur contrôlé à l'infini
    try:
        num, den = expr.as_numer_denom()
        if den != 1:
            nmin, nmax = encadrements(num, n)
            dmin, dmax = encadrements(den, n)
            s = signe_asymptotique(den, n)
            if s == 1:
                return (
                    sp.simplify(min_borne(nmin / dmax, nmax / dmax)),
                    sp.simplify(max_borne(nmin / dmin, nmax / dmin)),
                )
            if s == -1:
                return (
                    sp.simplify(min_borne(nmax / dmax, nmin / dmax)),
                    sp.simplify(max_borne(nmax / dmin, nmin / dmin)),
                )
    except:
        pass

    return expr, expr


# ============================================================
# Graphe (optionnel)
# ============================================================
def graphe(u_n, n, c):
    try:
        domaine = continuous_domain(u_n, n, S.Reals)
    except:
        domaine = S.Reals

    x_, y_ = [], []

    if isinstance(u_n, sp.Sum):
        expr = u_n.function
        (kk, n_min, _) = u_n.limits[0]
        f = sp.lambdify(kk, expr, "numpy")

        for N in range(0, 400):
            if domaine.contains(N):
                if N < int(n_min):
                    continue
                try:
                    s = 0.0
                    for j in range(int(n_min), N + 1):
                        s += float(f(j))
                    # On ajoute x et y ensemble uniquement si tout a réussi
                    x_.append(N)
                    y_.append(s)
                except:
                    pass
    else:
        f = sp.lambdify(n, u_n, "numpy")

        for N in range(0, 1500):
            if domaine.contains(N):
                try:
                    # IMPORTANT : on calcule d'abord y ; si ça échoue, on n'ajoute rien
                    y_val = float(f(N))
                    x_.append(N)
                    y_.append(y_val)
                except:
                    pass

    # Si rien n'est traçable, on évite de faire planter matplotlib
    if len(x_) == 0 or len(y_) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x_, y_, color=c, label=r"$" + sp.latex(u_n) + r"$")
    ax.set_xlabel("n")
    ax.set_ylabel("$" + sp.latex(u_n) + "$")
    ax.legend()
    st.pyplot(fig)




def diverge(u_n, n):
    try:
        lim = sp.limit_seq(u_n, n)

        if lim is not None and lim != 0:
            st.latex(
                r"\lim_{n \to \infty} " + sp.latex(u_n) + " = " + sp.latex(lim)
            )
            st.latex(
                r"\text{Le terme général ne tend pas vers } 0,\text{ la série est donc divergente.}"
            )
            return True

        return False

    except:
        return False






# ============================================================
# Positivité + décroissance (Leibniz)
# ============================================================
def positivite(u_n, n, n_min):
    """
    Vérifie (quand possible) u_n >= 0 pour n >= n_min, et affiche une justification.
    """
    try:
        x = sp.symbols("x", real=True)
        domaine = S.Reals.intersect(sp.Interval(n_min, sp.oo))
        ineq = sp.solve_univariate_inequality(
            sp.simplify(u_n.subs(n, x)) >= 0,
            x,
            relational=False,
            domain=domaine,
        )
        ok = (u_n.is_nonnegative is True) or (ineq != sp.EmptySet)
        if ok:
            msg = (
                r"\text{On vérifie que } "
                + sp.latex(u_n)
                + r"\ge 0 \text{ pour } n\ge "
                + sp.latex(n_min)
                + r"."
            )
            return True, msg
        return None, None
    except:
        if u_n.is_nonnegative is True:
            msg = r"\text{On a } " + sp.latex(u_n) + r"\ge 0."
            return True, msg
        return None, None


def decroissance(a_n, n, n_min):
    try:
        # Définition de la suite
        st.latex(
            r"\text{On considère la suite } (a_n)_{n\in\mathbb{N}} "
            r"\text{ définie par } a_n = " + sp.latex(a_n) + r"."
        )

        a_n1 = sp.simplify(a_n.subs(n, n + 1))

        # ============================================================
        # 1) Étude directe de a_{n+1} - a_n
        # ============================================================
        try:
            delta = sp.simplify(a_n1 - a_n)
            if delta.is_nonpositive:
                st.latex(
                    r"\text{Pour tout } n \ge " + sp.latex(n_min) + r",\quad "
                    r"a_{n+1}-a_n = "
                    + sp.latex(a_n1) + r" - " + sp.latex(a_n)
                    + r" = " + sp.latex(delta) + r"\le 0."
                )
                st.latex(
                    r"\text{Ainsi, la suite } (a_n) \text{ est décroissante à partir de } "
                    + sp.latex(n_min) + r"."
                )
                return True, None
        except:
            pass

        # ============================================================
        # 2) Étude par le quotient a_{n+1}/a_n
        # ============================================================
        try:
            q = sp.simplify(a_n1 / a_n)

            # On impose la positivité pour que la comparaison ait un sens
            pos_a_n, _ = positivite(a_n, n, n_min)
            pos_a_n1, _ = positivite(a_n1, n, n_min)

            if pos_a_n and pos_a_n1:
                test_q = sp.simplify(q - 1)
                if test_q.is_nonpositive:
                    st.latex(
                        r"\text{Pour tout } n \ge " + sp.latex(n_min) + r",\quad "
                        r"\frac{a_{n+1}}{a_n} = " + sp.latex(q) + r"\le 1."
                    )
                    st.latex(
                        r"\text{Comme } a_n \ge 0,\text{ la suite } (a_n) "
                        r"\text{ est décroissante à partir de } "
                        + sp.latex(n_min) + r"."
                    )
                    return True, None
        except:
            pass
        
        # ============================================================
        # 3) Étude par la fonction associée et sa dérivée
        # ============================================================
        try:
            x = sp.symbols('x', real=True)

            f = a_n.subs(n, x)

            st.latex(
                r"\text{On associe à la suite la fonction } "
                r"f : \mathbb{R} \to \mathbb{R} \text{ telle que } "
                r"f(n)=a_n \text{ pour tout entier } n."
            )
            st.latex(
                r"\text{On a donc, pour tout réel } x,\quad f(x) = "
                + sp.latex(f) + r"."
            )

            # Domaine de dérivabilité, restreint à x ≥ n_min
            domaine = continuous_domain(f, x, S.Reals)
            domaine = domaine.intersect(sp.Interval(n_min, sp.oo))

            st.latex(
                r"\text{La fonction } f \text{ est dérivable sur } "
                + sp.latex(domaine) + r"."
            )

            f_prime = sp.diff(f, x)
            st.latex(
                r"\text{Sa dérivée est donnée par } f'(x) = "
                + sp.latex(f_prime) + r"."
            )

            # --------------------------------------------------------
            # Test direct du signe de la dérivée
            # --------------------------------------------------------
            if f_prime.is_negative:
                st.latex(
                    r"\text{On a } f'(x) < 0 \text{ pour tout } x \in "
                    + sp.latex(domaine) + r"."
                )
                st.latex(
                    r"\text{La fonction } f \text{ est strictement décroissante sur ce domaine.}"
                )
                st.latex(
                    r"\text{Par conséquent, la suite } (a_n) \text{ est décroissante pour tout } n \ge "
                    + sp.latex(n_min) + r"."
                )
                return True, None

            if f_prime.is_nonpositive:
                st.latex(
                    r"\text{On a } f'(x) \le 0 \text{ pour tout } x \in "
                    + sp.latex(domaine) + r"."
                )
                st.latex(
                    r"\text{La fonction } f \text{ est décroissante sur ce domaine.}"
                )
                st.latex(
                    r"\text{Par conséquent, la suite } (a_n) \text{ est décroissante pour tout } n \ge "
                    + sp.latex(n_min) + r"."
                )
                return True, None

            # --------------------------------------------------------
            # 4) Étude complète du signe de f' par inéquation
            # --------------------------------------------------------
            try:
                domaine_signe = sp.solve_univariate_inequality(
                    f_prime <= 0, x, relational=False
                )

                domaine_utile = domaine.intersect(domaine_signe)

                if domaine_utile.sup == sp.oo:
                    st.latex(
                        r"\text{On étudie le signe de } f'(x) = "
                        + sp.latex(f_prime) + r"."
                    )
                    st.latex(
                        r"\text{On obtient } f'(x) \le 0 \text{ pour } x \in "
                        + sp.latex(domaine_utile) + r"."
                    )
                    st.latex(
                        r"\text{La fonction } f \text{ est donc décroissante sur cet intervalle.}"
                    )
                    st.latex(
                        r"\text{Il existe alors un entier } N \ge "
                        + sp.latex(n_min) + r" \text{ tel que pour tout } n \ge N,\ a_{n+1} \le a_n."
                    )
                    st.latex(
                        r"\text{Ainsi, la suite } (a_n) \text{ est décroissante à partir d'un certain rang.}"
                    )
                    return True, None
            except:
                pass

        except:
            pass

    except Exception as e:
        return None, f"Détail technique (décroissance) : {e}"

    return False, None


# ============================================================
# Alternance / extraction partie positive
# ============================================================
def detecte_alternance(expr, n):
    for p in expr.atoms(sp.Pow):
        base, exp = p.args
        if base == -1 and exp.has(n):
            return True
    return False


def extraire_partie_positive(expr, n, n_min):
    """
    Si expr = (-1)^n * q(n), renvoie q(n) si q(n) >= 0 (ou -q si q<=0).
    """
    for p in expr.atoms(sp.Pow):
        base, exp = p.args
        if base == -1 and exp.has(n):
            q = sp.simplify(expr / p)
            if q.is_nonnegative or (positivite(q, n, n_min)[0] is True):
                return sp.simplify(q)
            if q.is_nonpositive:
                return sp.simplify(-q)
            return None
    return None


# ============================================================
# DL : on n'affiche que l'équivalence (~) et la mention "d'après un DL"
# ============================================================
def dl_equivalent(u_n, n):
    try:
        x = sp.symbols("x")
        u_x = sp.simplify(u_n.subs(n, 1 / x))
        ser = sp.series(u_x, x, 0, 6)
        eq = sp.simplify(ser.as_leading_term(x).subs(x, 1 / n))

        if sp.simplify(eq - u_n).equals(0):
            return None

        st.latex(
            r"\text{On obtient } "
            + sp.latex(u_n)
            + r"\sim "
            + sp.latex(eq)
            + r"\ \text{(d'après un développement limité en } \frac{1}{n}\text{).}"
        )
        return eq
    except:
        return None


# ============================================================
# Critère de Riemann (1/n^p)
# ============================================================
def prepare_Riemann(expression, n):
    expression_approx = expression.evalf()
    facteur_val = facteur(expression_approx)
    if facteur_val == 0:
        facteur_val = 1
    expression_approx = expression_approx / facteur_val
    
    numerateur = expression_approx.as_numer_denom()[0]
    try:
        num = int(numerateur)
    except (TypeError, ValueError):
        return None
    
    dict = expression.as_numer_denom()[1].as_powers_dict()
    return num, dict

def test_Riemann(u_n, n, n_min):
    S_n = sp.Sum(u_n.subs(n, k), (k, n_min, n))

    P = prepare_Riemann(u_n, n)
    if P is None:
        return None

    num, d = P
    if num == 1 and len(d) == 1 and list(d.keys())[0] == n:
        exposant = d[n]
        
        if exposant > 1:
            st.latex(
                r"\text{Comme } "
                + sp.latex(exposant)
                + r">1,\ \sum \frac{1}{n^{"
                + sp.latex(exposant)
                + r"}} \text{ converge par le critère de Riemann. Donc } "
                + sp.latex(S_n)
                + r"\text{ converge.}"
            )
            return True
        else:
            st.latex(
                r"\text{Comme } "
                + sp.latex(exposant)
                + r"\le 1,\ \sum \frac{1}{n^{"
                + sp.latex(exposant)
                + r"}} \text{ diverge (critère de Riemann). Donc } "
                + sp.latex(S_n)
                + r"\text{ diverge.}"
            )
            return False

    return None


# ============================================================
# Bertrand (1/(n (log n)^a))
# ============================================================
def facteur(expr):
    c, _ = expr.as_coeff_Mul()
    return c


def test_bertrand(u_n, n, n_min):
    S_n = sp.Sum(u_n.subs(n, k), (k, n_min, n))

    u_s = sp.simplify(u_n / facteur(u_n))
    num, den = u_s.as_numer_denom()

    # On ne traite que num = 1
    try:
        if int(num) != 1:
            return None
    except:
        return None

    facs = list(den.args) if den.is_Mul else [den]
    if n not in facs:
        return None

    others = [f for f in facs if f != n]
    exposant_log = S.Zero

    for f in others:
        if f == sp.log(n):
            exposant_log += 1
            continue
        if f.is_Pow:
            base, exp = f.as_base_exp()
            if base == sp.log(n):
                exposant_log += exp
                continue
        return None

    st.latex(
        r"\text{On reconnaît } "
        + sp.latex(u_n)
        + r"=\frac{1}{n(\log n)^{"
        + sp.latex(exposant_log)
        + r"}}."
    )

    if exposant_log > 1:
        st.latex(
            r"\text{Comme } "
            + sp.latex(exposant_log)
            + r">1,\ \sum \frac{1}{n(\log n)^{"
            + sp.latex(exposant_log)
            + r"}} \text{ converge (critère de Bertrand). Donc } "
            + sp.latex(S_n)
            + r"\text{ converge.}"
        )
        return True

    st.latex(
        r"\text{Comme } "
        + sp.latex(exposant_log)
        + r"\le 1,\ \sum \frac{1}{n(\log n)^{"
        + sp.latex(exposant_log)
        + r"}} \text{ diverge (critère de Bertrand). Donc } "
        + sp.latex(S_n)
        + r"\text{ diverge.}"
    )
    return False


# ============================================================
# Critère des logs itérés (famille n^p * log * loglog * ...)
# ============================================================
def _is_iterated_log_of_n(expr, n, max_depth=8):
    """
    True si expr = log(log(...log(n)...)) (au moins 1 log).
    """
    depth = 0
    t = expr
    while depth < max_depth:
        if t == n:
            return depth >= 1
        if t.func != sp.log or len(t.args) != 1:
            return False
        t = t.args[0]
        depth += 1
    return False


def parse_polylog_term(u_n, n):
    """
    Reconnaît :  1 / ( n^p * Π (log^{∘i}(n))^{beta_i} )
    Renvoie : (p, [(log_i, beta_i) trié par profondeur])
    """
    u_s = sp.simplify(u_n)
    num, den = u_s.as_numer_denom()
    if num != 1:
        return None

    den_factors = list(den.args) if den.is_Mul else [den]

    exposant_n = S.Zero
    logs = []  # (depth, base, expo)

    for f in den_factors:
        # n^p
        if f == n:
            exposant_n += 1
            continue
        if f.is_Pow and f.base == n:
            exposant_n += f.exp
            continue

        # log^{...}
        base, expo = None, None
        if f == sp.log(n):
            base, expo = sp.log(n), 1
        elif f.is_Pow:
            b, e = f.as_base_exp()
            if _is_iterated_log_of_n(b, n):
                base, expo = b, e

        if base is None:
            return None

        # profondeur
        depth = 0
        t = base
        while t != n:
            t = t.args[0]
            depth += 1
        logs.append((depth, base, expo))

    logs.sort(key=lambda x: x[0])
    return exposant_n, [(b, e) for _, b, e in logs]


def test_polylog(u_n, n, n_min):
    """
    Décision standard :
    - si p>1 : converge
    - si p<1 : diverge
    - si p=1 : on regarde les exposants des logs itérés :
        premier exposant != 1 : >1 => converge, <=1 => diverge
        tous = 1 => diverge
    """
    S_n = sp.Sum(u_n.subs(n, k), (k, n_min, n))
    parsed = parse_polylog_term(u_n, n)
    if parsed is None:
        return None

    exposant_n, logs = parsed

    # On annonce la forme (sans lettre non définie : on affiche l'exposant réel rencontré)
    st.latex(
        r"\text{On met } "
        + sp.latex(u_n)
        + r"\text{ sous la forme } \frac{1}{n^{"
        + sp.latex(exposant_n)
        + r"}}\cdot\frac{1}{\prod (\log^{\circ i}(n))^{\beta_i}}\text{ (avec les exposants affichés ci-dessous).}"
    )

    # p>1 / p<1 / p=1
    if exposant_n > 1:
        st.latex(
            r"\text{Comme } "
            + sp.latex(exposant_n)
            + r">1,\ \sum \frac{1}{n^{"
            + sp.latex(exposant_n)
            + r"}} \text{ converge (Riemann), et } "
            + sp.latex(u_n)
            + r"\le \frac{1}{n^{"
            + sp.latex(exposant_n)
            + r"}}\text{ pour } n\ge 2.\ \text{Donc } "
            + sp.latex(S_n)
            + r"\text{ converge (comparaison).}"
        )
        return True

    if exposant_n < 1:
        st.latex(
            r"\text{Comme } "
            + sp.latex(exposant_n)
            + r"<1,\ \sum \frac{1}{n^{"
            + sp.latex(exposant_n)
            + r"}} \text{ diverge (Riemann), et } "
            + sp.latex(u_n)
            + r"\ge \frac{1}{n^{"
            + sp.latex(exposant_n)
            + r"}}\text{ pour } n\ge 2.\ \text{Donc } "
            + sp.latex(S_n)
            + r"\text{ diverge (comparaison).}"
        )
        return False

    # cas critique p=1
    if logs == []:
        st.latex(r"\text{Ici } " + sp.latex(u_n) + r"=\frac{1}{n},\ \sum \frac{1}{n} \text{ diverge. Donc } " + sp.latex(S_n) + r"\text{ diverge.}")
        return False

    # on parcourt les logs dans l'ordre
    produit_critique = []
    for base, expo in logs:
        produit_critique.append(base)

        st.latex(
            r"\text{On considère l'exposant de } "
            + sp.latex(base)
            + r"\text{ : } \beta="
            + sp.latex(expo)
            + r"."
        )

        if sp.simplify(expo - 1).equals(0):
            continue

        if expo > 1:
            st.latex(
                r"\text{Au premier rang où } \beta\neq 1,\ \beta="
                + sp.latex(expo)
                + r">1,\ \text{donc } "
                + sp.latex(S_n)
                + r"\text{ converge (critère des logs itérés).}"
            )
            return True
        else:
            st.latex(
                r"\text{Au premier rang où } \beta\neq 1,\ \beta="
                + sp.latex(expo)
                + r"\le 1,\ \text{donc } "
                + sp.latex(S_n)
                + r"\text{ diverge (critère des logs itérés).}"
            )
            return False

    st.latex(
        r"\text{Tous les exposants rencontrés valent } 1,\ \text{on est dans un cas critique de type } "
        r"\sum \frac{1}{n\log n\log\log n\cdots} \text{ qui diverge. Donc } "
        + sp.latex(S_n)
        + r"\text{ diverge.}"
    )
    return False


# ============================================================
# d'Alembert (ratio)
# ============================================================
def test_Alembert(u_n, n, n_min):
    S_n = sp.Sum(u_n.subs(n, k), (k, n_min, n))
    u_s = sp.simplify(u_n)

    if u_s == 0:
        return None

    try:
        u_n1 = sp.simplify(u_s.subs(n, n + 1))
        q = sp.simplify(sp.Abs(u_n1 / u_s))
        L = sp.limit_seq(q, n)
        if L is None or L.is_real is False:
            return None

        if L < 1:
            st.latex(
                r"\text{On calcule } \left|\frac{u_{n+1}}{u_n}\right|="
                + sp.latex(q)
                + r"\ \text{et } \lim_{n\to\infty}\left|\frac{u_{n+1}}{u_n}\right|="
                + sp.latex(L)
                + r"<1."
            )
            st.latex(r"\text{Donc } " + sp.latex(S_n) + r"\text{ converge (critère d'Alembert).}")
            return True

        if L > 1:
            st.latex(
                r"\text{On calcule } \left|\frac{u_{n+1}}{u_n}\right|="
                + sp.latex(q)
                + r"\ \text{et } \lim_{n\to\infty}\left|\frac{u_{n+1}}{u_n}\right|="
                + sp.latex(L)
                + r">1."
            )
            st.latex(r"\text{Donc } " + sp.latex(S_n) + r"\text{ diverge (critère d'Alembert).}")
            return False

    except:
        pass

    return None


# ============================================================
# Cauchy (racine n-ième)
# ============================================================
def test_Cauchy_racine(u_n, n, n_min):
    S_n = sp.Sum(u_n.subs(n, k), (k, n_min, n))
    try:
        v = sp.simplify(sp.Abs(u_n))
        r = sp.simplify(v ** (1 / n))
        L = sp.limit_seq(r, n)
        if L is None or L.is_real is False:
            return None

        if L < 1:
            st.latex(
                r"\text{On calcule } \sqrt[n]{|u_n|}="
                + sp.latex(r)
                + r"\ \text{et } \lim_{n\to\infty}\sqrt[n]{|u_n|}="
                + sp.latex(L)
                + r"<1."
            )
            st.latex(r"\text{Donc } " + sp.latex(S_n) + r"\text{ converge absolument (critère de Cauchy).}")
            return True

        if L > 1:
            st.latex(
                r"\text{On calcule } \sqrt[n]{|u_n|}="
                + sp.latex(r)
                + r"\ \text{et } \lim_{n\to\infty}\sqrt[n]{|u_n|}="
                + sp.latex(L)
                + r">1."
            )
            st.latex(r"\text{Donc } " + sp.latex(S_n) + r"\text{ diverge (critère de Cauchy).}")
            return False

    except:
        pass

    return None


# ============================================================
# Leibniz
# ============================================================
def test_Leibniz(u_n, n, n_min):
    S_n = sp.Sum(u_n.subs(n, k), (k, n_min, n))
    u_s = sp.simplify(u_n)

    if not (detecte_alternance(u_s, n) or detecte_alternance(u_s, n + 1)):
        return None

    a_n = extraire_partie_positive(u_s, n, n_min)
    
    if a_n is None:
        return None
    
    lim = sp.limit_seq(a_n, n)
    if lim != 0:
        return None
    
    ok_pos, msg_pos = positivite(a_n, n, n_min)
    if ok_pos is not True:
        return None

    ok_dec, msg_dec = decroissance(a_n, n, n_min)
    if ok_dec is not True:
        return None

    

    st.latex(msg_pos)
    st.latex(msg_dec)
    st.latex(r"\text{De plus } \lim_{n\to\infty} " + sp.latex(a_n) + r"=0.")
    st.latex(r"\text{Donc } " + sp.latex(S_n) + r"\text{ converge (critère de Leibniz).}")
    return True


# ============================================================
# Comparaison (générale + cas bornés n^p +/- borné)
# ============================================================
def _detect_main_power(den, n):
    """
    Si den contient un terme n ou n^p (p réel), renvoie ce terme (sympy).
    Priorité : n^p si présent, sinon n.
    """
    if not den.is_Add:
        return None
    # cherche n^p
    for t in den.args:
        if t.is_Pow and t.base == n:
            return t
    # sinon n
    for t in den.args:
        if t == n:
            return n
    return None


def _comparison_den_power_plus_bounded(u_s, n, n_min):
    """
    Cas u_s = 1/(n^p + b(n)) avec b(n) bornée via encadrements.
    Décision (si p est comparable) :
    - p>1 => convergence (comparaison à 1/n^p)
    - p=1 => divergence (comparaison à 1/(n+C) ~ 1/n)
    - 0<p<1 => divergence (comparaison à 1/n^p)
    - p<=0 => les termes ne tendent pas vers 0 => divergence
    """
    num, den = u_s.as_numer_denom()
    if num != 1 or not den.is_Add:
        return None

    main = _detect_main_power(den, n)
    if main is None:
        return None

    perturb = sp.simplify(den - main)
    bmin, bmax = encadrements(perturb, n)

    # on n'accepte que des bornes numériques (sin/cos/(-1)^n etc => OK)
    if not (bmin.is_Number and bmax.is_Number):
        return None

    M = sp.Max(sp.Abs(bmin), sp.Abs(bmax))
    p = S.One if main == n else main.exp

    # on affiche l'écriture + encadrement
    st.latex(
        r"\text{On écrit } "
        + sp.latex(den)
        + r"="
        + sp.latex(main)
        + r"+"
        + sp.latex(perturb)
        + r"\ \text{avec } "
        + sp.latex(bmin)
        + r"\le "
        + sp.latex(perturb)
        + r"\le "
        + sp.latex(bmax)
        + r"."
    )

    # p<=0 : termes ne vont pas vers 0
    if p <= 0:
        st.latex(
            r"\text{Ici l'exposant sur } n \text{ vaut } "
            + sp.latex(p)
            + r"\le 0,\ \text{donc } "
            + sp.latex(main)
            + r"\text{ ne tend pas vers } +\infty.\ \text{Ainsi } "
            + sp.latex(u_s)
            + r"\text{ ne tend pas vers }0,\ \text{donc la série diverge.}"
        )
        return False

    # p>1 : convergence
    if p > 1:
        # pour n>=2, on a n^p - M >= (1/2)n^p dès que n^p >= 2M
        # (on reste sur une justification lisible, sans introduire une lettre non expliquée)
        st.latex(
            r"\text{Pour } n \text{ assez grand, on a } "
            + sp.latex(main)
            + r"-"
            + sp.latex(M)
            + r"\ge \frac{1}{2}"
            + sp.latex(main)
            + r"\ \Rightarrow\ "
            + sp.latex(den)
            + r"\ge \frac{1}{2}"
            + sp.latex(main)
            + r"."
        )
        st.latex(
            r"\text{Donc } "
            + sp.latex(u_s)
            + r"\le \frac{2}{"
            + sp.latex(main)
            + r"}=\frac{2}{n^{"
            + sp.latex(p)
            + r"}}."
        )
        st.latex(
            r"\text{Or } \sum \frac{1}{n^{"
            + sp.latex(p)
            + r"}} \text{ converge (Riemann), donc } \sum "
            + sp.latex(u_s)
            + r"\text{ converge (comparaison).}"
        )
        return True

    # p=1 : divergence
    if sp.simplify(p - 1).equals(0):
        st.latex(
            r"\text{On a } "
            + sp.latex(den)
            + r"\le "
            + sp.latex(main)
            + r"+"
            + sp.latex(M)
            + r"\ \Rightarrow\ "
            + sp.latex(u_s)
            + r"\ge \frac{1}{n+"
            + sp.latex(M)
            + r"}."
        )
        st.latex(
            r"\text{Or } \sum \frac{1}{n+"
            + sp.latex(M)
            + r"} \text{ diverge (comparable à } \sum \frac{1}{n}\text{), donc } \sum "
            + sp.latex(u_s)
            + r"\text{ diverge (comparaison).}"
        )
        return False

    # 0<p<1 : divergence
    if p < 1:
        st.latex(
            r"\text{On a } "
            + sp.latex(den)
            + r"\le "
            + sp.latex(main)
            + r"+"
            + sp.latex(M)
            + r"\ \Rightarrow\ "
            + sp.latex(u_s)
            + r"\ge \frac{1}{n^{"
            + sp.latex(p)
            + r"}+"
            + sp.latex(M)
            + r"}."
        )
        st.latex(
            r"\text{Pour } n \text{ assez grand, } n^{"
            + sp.latex(p)
            + r"}+"
            + sp.latex(M)
            + r"\le 2n^{"
            + sp.latex(p)
            + r"}\ \Rightarrow\ "
            + sp.latex(u_s)
            + r"\ge \frac{1}{2}\cdot\frac{1}{n^{"
            + sp.latex(p)
            + r"}}."
        )
        st.latex(
            r"\text{Or } \sum \frac{1}{n^{"
            + sp.latex(p)
            + r"}} \text{ diverge (Riemann, exposant }\le 1\text{), donc } \sum "
            + sp.latex(u_s)
            + r"\text{ diverge (comparaison).}"
        )
        return False

    return None


def test_comparaison(u_n, n, n_min):
    S_n = sp.Sum(u_n.subs(n, k), (k, n_min, n))
    u_s = sp.simplify(u_n)

    # ------------------------------------------------------------
    # Cas : 1/(log(n))^q (sans n) => divergence par comparaison à 1/n
    # ------------------------------------------------------------
    try:
        num, den = u_s.as_numer_denom()
        if num == 1:
            if den == sp.log(n):
                st.latex(
                    r"\text{Pour } n\ge 3,\ \log(n)<n\ \Rightarrow\ "
                    + sp.latex(u_s)
                    + r"=\frac{1}{\log(n)}\ge \frac{1}{n}."
                )
                st.latex(
                    r"\text{Or } \sum \frac{1}{n} \text{ diverge, donc } "
                    + sp.latex(S_n)
                    + r"\text{ diverge (comparaison).}"
                )
                return False

            if den.is_Pow:
                base, exp = den.as_base_exp()
                if base == sp.log(n) and exp.is_positive:
                    st.latex(
                        r"\text{Pour } n\ge 3,\ \log(n)<n\ \Rightarrow\ "
                        + sp.latex(u_s)
                        + r"=\frac{1}{(\log(n))^{"
                        + sp.latex(exp)
                        + r"}}\ge \frac{1}{n}."
                    )
                    st.latex(
                        r"\text{Or } \sum \frac{1}{n} \text{ diverge, donc } "
                        + sp.latex(S_n)
                        + r"\text{ diverge (comparaison).}"
                    )
                    return False
    except:
        pass

    # ------------------------------------------------------------
    # Cas : 1/(n^p + borné)  (sin/cos/(-1)^n/combinaisons bornées)
    # ------------------------------------------------------------
    try:
        num, den = u_s.as_numer_denom()
        if num == 1 and den.is_Add:
            r = _comparison_den_power_plus_bounded(u_s, n, n_min)
            if r is not None:
                return r
    except:
        pass

    # ------------------------------------------------------------
    # Sinon : MAJORATION automatique uniquement (pour conclure la convergence).
    # (On n'utilise pas une minoration automatique pour conclure la divergence.)
    # ------------------------------------------------------------
    try:
        a, b = encadrements(u_n, n)
        max_un = max_borne(a, b)

        # majoration -> convergence
        try:
            test_sup = critere(max_un, n, n_min)
            if test_sup is True:
                st.latex(
                    r"\text{On a } "
                    + sp.latex(u_n)
                    + r"\le "
                    + sp.latex(max_un)
                    + r"\ \text{pour } n\ge "
                    + sp.latex(n_min)
                    + r"."
                )
                st.latex(
                    r"\text{Comme } \sum "
                    + sp.latex(max_un)
                    + r"\text{ converge, } "
                    + sp.latex(S_n)
                    + r"\text{ converge (comparaison).}"
                )
                return True
        except:
            pass

    except:
        pass

    return None


# ============================================================
# Équivalence
# ============================================================
def test_equivalence(u_n, n, n_min):
    S_n = sp.Sum(u_n.subs(n, k), (k, n_min, n))
    eq = dl_equivalent(u_n, n)
    if eq is None:
        return None

    C = critere(eq, n, n_min)
    if C is True:
        st.latex(r"\text{Donc } " + sp.latex(S_n) + r"\text{ converge par équivalence.}")
        return True
    if C is False:
        st.latex(r"\text{Donc } " + sp.latex(S_n) + r"\text{ diverge par équivalence.}")
        return False

    return None


# ============================================================
# Convergence absolue
# ============================================================
def convergence_absolue(u_n, n, n_min):
    v = sp.Abs(u_n)

    # critères "propres" sur |u_n|, sans sur-traitement
    for f in (test_Riemann, test_polylog, test_bertrand, test_Cauchy_racine, test_Alembert):
        r = f(v, n, n_min)
        if r is True:
            st.latex(r"\text{Donc la série } \sum " + sp.latex(u_n) + r"\text{ converge absolument.}")
            return True
        if r is False:
            # divergence de la valeur absolue ne conclut pas sur la série initiale
            return None

    return None

def divergence_grossiere(u_n , n) :
    if diverge(u_n, n):
            lim = sp.limit_seq(u_n, n)
            if lim is not None and not getattr(lim, "is_zero", False):
                st.latex(r"\text{La condition nécessaire n’est pas vérifiée : divergence grossiere.}")
                return True
    return False






# ============================================================
# Orchestrateur
# ============================================================
def critere(u_n, n, n_min):
    if not divergence_grossiere(u_n, n):
        # protection contre appels circulaires
        if not hasattr(critere, "_pile"):
            critere._pile = []

        u_s = sp.simplify(u_n)
        for old in critere._pile:
            if sp.simplify(old - u_s).equals(0):
                return None

        critere._pile.append(u_s)

        try:
            # ordre stratégique : alternée -> Riemann -> logs itérés -> Bertrand -> racine -> ratio -> comparaison -> équivalence
            for f in (
                test_Leibniz,
                test_Riemann,
                test_polylog,
                test_bertrand,
                test_Cauchy_racine,
                test_Alembert,
                test_comparaison,
            ):
                r = f(u_n, n, n_min)
                if r is not None:
                    return r

            # absolue (si ça conclut)
            if convergence_absolue(u_n, n, n_min) is True:
                return True

            # équivalence en dernier recours
            r = test_equivalence(u_n, n, n_min)
            if r is not None:
                return r

            return None

        finally:
            critere._pile.pop()


import numpy as np
import sympy as sp
from sympy import S
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify


# =============================================================
# 0) Faux Streamlit (fallback si jamais "st" n'existe pas déjà)
#    IMPORTANT : dans TON projet, "st" existe déjà (core SymPy).
# =============================================================
try:
    st  # noqa: F821
except NameError:
    import io, base64

    class _StreamlitCompat:
        def __init__(self):
            self._out = []

        def set_out(self, out):
            self._out = out

        def latex(self, msg):
            self._out.append({"type": "latex", "content": msg})

        def info(self, msg):
            self._out.append({"type": "info", "content": str(msg)})

        def error(self, msg):
            self._out.append({"type": "error", "content": str(msg)})

        def pyplot(self, fig):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("utf-8")
            self._out.append({"type": "plot", "content": b64})

    st = _StreamlitCompat()


# =============================================================
# 1) LaTeX : messages propres
# =============================================================
def _latex_rouge(msg: str) -> str:
    safe = (msg or "").replace("\\", r"\textbackslash ").replace("_", r"\_")
    return r"\textcolor{red}{\text{" + safe + r"}}"


def _latex_info(msg: str) -> str:
    safe = (msg or "").replace("\\", r"\textbackslash ").replace("_", r"\_")
    return r"\text{" + safe + r"}"


def _latex_intervalle(intervalle) -> str:
    if intervalle == S.Reals:
        return r"\mathbb{R}"
    if intervalle == S.EmptySet:
        return r"\varnothing"
    try:
        return sp.latex(intervalle)
    except Exception:
        return r"\varnothing"


# =============================================================
# 2) Parsing intervalle depuis payload (compatible app.js)
# =============================================================
def _parse_interval_payload(data):
    borne_gauche = (data.get("interval_left") or "").strip()
    borne_droite = (data.get("interval_right") or "").strip()
    ferme_gauche = bool(data.get("interval_left_closed", False))
    ferme_droite = bool(data.get("interval_right_closed", False))

    if not borne_gauche and not borne_droite:
        return S.Reals

    def _vers_borne(texte):
        texte = (texte or "").strip()
        if not texte:
            return None
        if texte in ("oo", "+oo", "∞", "+∞"):
            return sp.oo
        if texte in ("-oo", "-∞"):
            return -sp.oo
        try:
            return sp.sympify(texte)
        except Exception:
            return None

    a = _vers_borne(borne_gauche)
    b = _vers_borne(borne_droite)

    if a is None:
        a = -sp.oo
    if b is None:
        b = sp.oo

    if a == -sp.oo and b == sp.oo:
        return S.Reals

    left_open = True if a == -sp.oo else (not ferme_gauche)
    right_open = True if b == sp.oo else (not ferme_droite)

    try:
        if (a.is_real is True) and (b.is_real is True):
            if sp.simplify(a - b).is_positive is True:
                return S.EmptySet
            if sp.simplify(a - b).equals(0):
                if left_open or right_open:
                    return S.EmptySet
    except Exception:
        pass

    if a == -sp.oo:
        return sp.Interval(-sp.oo, b, left_open=True, right_open=right_open)
    if b == sp.oo:
        return sp.Interval(a, sp.oo, left_open=left_open, right_open=True)

    return sp.Interval(a, b, left_open=left_open, right_open=right_open)


def _intervalle_est_vide(intervalle):
    try:
        return intervalle == S.EmptySet
    except Exception:
        return False


# =============================================================
# 3) Limite simple (suite/série) : robuste
# =============================================================
def _limite_simple(expr, n):
    try:
        L = sp.limit_seq(expr, n)
        if L is None or isinstance(L, sp.Limit):
            return None, False
        return sp.simplify(L), True
    except Exception:
        return None, False


# =============================================================
# 4) Numérique robuste : eval vecteur
# =============================================================
def _eval_vecteur(x, expr, xs):
    if expr is None:
        return np.full(xs.shape, np.nan, dtype=float)
    try:
        f_num = sp.lambdify(x, expr, "numpy")
        with np.errstate(all="ignore"):
            y = f_num(xs)
        if np.isscalar(y):
            return np.full(xs.shape, float(y), dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != xs.size:
            return np.full(xs.shape, np.nan, dtype=float)
        return y
    except Exception:
        return np.full(xs.shape, np.nan, dtype=float)


def _bornes_intervalle_pour_graphe(intervalle):
    if intervalle == S.Reals:
        return -5.0, 5.0
    if isinstance(intervalle, sp.Interval):
        try:
            xmin = float(sp.N(intervalle.inf))
            xmax = float(sp.N(intervalle.sup))
            if (not np.isfinite(xmin)) or (not np.isfinite(xmax)) or (xmin == xmax):
                return -5.0, 5.0
            if (xmin == -np.inf) or (xmax == np.inf):
                return -5.0, 5.0
            return xmin, xmax
        except Exception:
            return -5.0, 5.0
    return -5.0, 5.0


# =============================================================
# 5) Graphes suites de fonctions / séries de fonctions
# =============================================================
def _plot_famille_et_limite(f_n, f_lim, x, intervalle, n, n_min=1, n_max=20):
    def _remplacer_n(expr, valeur):
        try:
            subs_map = {s: valeur for s in expr.free_symbols if getattr(s, "name", "") == "n"}
            if subs_map:
                return sp.simplify(expr.subs(subs_map))
            return sp.simplify(expr.subs(n, valeur))
        except Exception:
            return expr

    xmin, xmax = _bornes_intervalle_pour_graphe(intervalle)
    xs = np.linspace(xmin, xmax, 1600)
    fig, ax = plt.subplots(figsize=(18, 5))

    cmap = plt.get_cmap("tab20")
    couleurs = [cmap(i) for i in range(20)]

    k = 0
    for i in range(int(n_min), int(n_max) + 1):
        f_i = _remplacer_n(f_n, i)
        y_i = _eval_vecteur(x, f_i, xs)
        ax.plot(xs, y_i, color=couleurs[k % 20], linewidth=1.6, alpha=0.95, label=rf"$f_{{{i}}}(x)$")
        k += 1

    if f_lim is not None:
        y_lim = _eval_vecteur(x, f_lim, xs)
        ax.plot(xs, y_lim, color="black", linewidth=3.0, label=r"$f(x)=\lim_{n\to\infty} f_n(x)$")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)


def _plot_series_partielles(u_n, x, intervalle, n, n_min=1, N_max=20):
    xmin, xmax = _bornes_intervalle_pour_graphe(intervalle)
    xs = np.linspace(xmin, xmax, 1400)

    N_max = int(N_max)
    n_min = int(n_min)
    if N_max < n_min:
        N_max = n_min

    fig, ax = plt.subplots(figsize=(18, 5))
    cmap = plt.get_cmap("tab20")

    for N in range(n_min, N_max + 1):
        Sx = np.zeros_like(xs, dtype=float)
        for k in range(n_min, N + 1):
            try:
                terme_k = sp.simplify(u_n.subs(n, k))
                Sx += _eval_vecteur(x, terme_k, xs)
            except Exception:
                pass
        ax.plot(xs, Sx, color=cmap((N - 1) % 20), linewidth=1.8, label=rf"$S_{{{N}}}(x)$")

    ax.set_xlabel("x")
    ax.set_ylabel("Sommes partielles")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)


# =============================================================
# 6bis) Convergence simple / uniforme / normale (suites et séries de fonctions)
#       Les fonctions _cv_* ne font aucun st.latex : elles renvoient
#       des verdicts et des lignes LaTeX, affichées ensuite dans l'ordre.
# =============================================================
from sympy.calculus.accumulationbounds import AccumulationBounds as _AccB


def _cv_avec_delai(fn, secondes):
    """Exécute fn() avec un délai maximal (aussi hors du thread principal). Renvoie None en cas d'échec."""
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


def _cv_capture(fn, *args, **kwargs):
    """Appelle fn en capturant les st.latex émis. Renvoie (résultat, [lignes LaTeX])."""
    ancien = st._out
    tmp = []
    st.set_out(tmp)
    try:
        res = _cv_avec_delai(lambda: fn(*args, **kwargs), 15)
    finally:
        st.set_out(ancien)
    lignes = [m["content"] for m in tmp if m.get("type") == "latex" and m.get("content")]
    return res, lignes


def _cv_txt(s):
    return r"\text{" + s + r"}"


def _cv_titre(s):
    return r"\textbf{" + s + r"}"


def _cv_extremites(I):
    if I == S.Reals:
        return -sp.oo, sp.oo, True, True
    return I.start, I.end, I.left_open, I.right_open


def _cv_est_segment(I):
    """Vrai si I = [a, b] avec a, b finis."""
    try:
        return isinstance(I, sp.Interval) and I.is_closed and I.start.is_finite and I.end.is_finite
    except Exception:
        return False


def _cv_decoupe(I, points):
    """Découpe I aux points donnés : liste ordonnée d'Interval et de FiniteSet."""
    a, b, lo, ro = _cv_extremites(I)
    pts = []
    for p in points:
        try:
            if p.is_real and I.contains(p) == S.true:
                pts.append(p)
        except Exception:
            pass
    pts = sorted(set(pts), key=lambda p: float(p))
    morceaux = []
    gauche, gauche_ouvert = a, lo
    for p in pts:
        if p == gauche and not gauche_ouvert:
            morceaux.append(sp.FiniteSet(p))
        else:
            seg = sp.Interval(gauche, p, gauche_ouvert, True)
            if seg != S.EmptySet:
                morceaux.append(seg)
            morceaux.append(sp.FiniteSet(p))
        gauche, gauche_ouvert = p, True
    fin = sp.Interval(gauche, b, gauche_ouvert, ro)
    if fin != S.EmptySet:
        morceaux.append(fin)
    return morceaux


def _cv_echantillons(piece):
    """Quelques points de test à l'intérieur d'un intervalle (ou le point lui-même)."""
    if isinstance(piece, sp.FiniteSet):
        return list(piece.args)
    a, b = piece.start, piece.end
    if a == -sp.oo and b == sp.oo:
        return [sp.Integer(-3), sp.Integer(2), sp.Integer(4)]
    if a == -sp.oo:
        return [b - 1, b - 2, b - 5]
    if b == sp.oo:
        return [a + 1, a + 2, a + 5]
    L = b - a
    return [a + L / 4, a + L / 2, a + 3 * L / 4]


def _cv_lim_n(expr, n):
    """Limite en n. Renvoie (statut, L) avec statut 'cv', 'div' (limite infinie),
    'osc' (pas de limite) ou 'ind' (SymPy ne conclut pas)."""
    for methode in (lambda e: sp.limit_seq(e, n), lambda e: sp.limit(e, n, sp.oo)):
        try:
            L = methode(expr)
        except Exception:
            continue
        if L is None or isinstance(L, sp.Limit):
            continue
        if isinstance(L, _AccB):
            try:
                if L.min != L.max:
                    return "osc", L
            except Exception:
                pass
            continue
        if L.has(n) or L is sp.nan or L == sp.nan:
            continue
        if L in (sp.oo, -sp.oo, sp.zoo):
            return "div", L
        return "cv", sp.simplify(L)
    return "ind", None


def _cv_sep_borne(expr, n):
    """expr = r * T, T produit de facteurs bornés (sin, cos, (-1)^n). Renvoie (r, a_du_borne)."""
    r = sp.Integer(1)
    borne = False
    for f in sp.Mul.make_args(sp.sympify(expr)):
        base = f.base if f.is_Pow else f
        est_trig = getattr(base, "func", None) in (sp.sin, sp.cos)
        est_alt = f.is_Pow and f.base == -1
        if f.has(n) and (est_alt or (est_trig and (not f.is_Pow or (f.exp.is_Integer and f.exp > 0)))):
            borne = True
            continue
        r = r * f
    return r, borne


def _cv_decomp_geometrique(expr, n, x):
    """Cherche expr = q(n, x) * b(x)**n. Renvoie (b, q) ou None."""
    try:
        expr = sp.simplify(expr)
        bases = []
        for p in expr.atoms(sp.Pow):
            if p.exp.has(n) and not p.base.has(n):
                e = sp.expand(p.exp)
                c = sp.diff(e, n)
                if c.has(n) or c.has(x) or (not c.is_Integer) or c == 0:
                    return None
                d = sp.simplify(e - c * n)
                if not d.is_Integer:
                    return None
                bases.append(p.base ** c)
        for ex in expr.atoms(sp.exp):
            arg = sp.expand(ex.args[0])
            if arg.has(n):
                c = sp.diff(arg, n)
                if c.has(n):
                    return None
                bases.append(sp.exp(c))
        if not bases:
            return None
        b = sp.simplify(sp.Mul(*bases))
        if not b.has(x) and sp.Abs(b) == 1:
            return None  # base constante de module 1 ((-1)^n) : traitée par la voie générale
        q = sp.simplify(expr / b ** n)
        for p in q.atoms(sp.Pow):
            if p.exp.has(n) and not p.base.has(n):
                return None
        for ex in q.atoms(sp.exp):
            if ex.args[0].has(n):
                return None
        return b, q
    except Exception:
        return None


def _cv_terme_dominant(r, n):
    """r rationnelle en n : renvoie (k, c) avec r ~ c(x) n^k."""
    num, den = sp.fraction(sp.together(r))
    pn, pd = sp.Poly(num, n), sp.Poly(den, n)
    if pn.is_zero:
        return None
    return pn.degree() - pd.degree(), sp.simplify(pn.LC() / pd.LC())


def _cv_val(expr, n, x, xv, N):
    try:
        v = sp.N(expr.subs({n: N, x: xv}), 25)
        if v.is_real:
            return float(v)
    except Exception:
        pass
    return None


def _cv_verif_numerique(f, L, n, x, piece):
    """Garde-fou : f_N(x) est proche de L(x) pour quelques x de la pièce et N grand."""
    for xv in _cv_echantillons(piece):
        a = _cv_val(f, n, x, xv, 4000)
        b = _cv_val(L, n, x, xv, 4000) if L.has(x) else _cv_val(L, n, x, xv, 1)
        if a is None or b is None:
            return False
        if abs(a - b) > 5e-2 * (1 + abs(b)):
            return False
    return True


def _cv_txt_borne(f, r0, n):
    """Phrase concrète : le facteur borné (sin(nx), (-1)^n, ...) et le terme qui tend vers 0."""
    try:
        bf = sp.simplify(f / r0)
    except Exception:
        bf = None
    lat_r = sp.latex(r0)
    if bf is None:
        return r"\left|f_n(x)\right|\le \left|" + lat_r + r"\right|\longrightarrow 0"
    return (r"\left|" + sp.latex(bf) + r"\right|\le 1 \text{ pour tous } n, x \text{, donc } \left|f_n(x)\right|\le \left|" + lat_r
            + r"\right|\underset{n\to+\infty}{\longrightarrow} 0")


def _cv_pw(b, n):
    """b^n en LaTeX, parenthèses seulement si nécessaire."""
    lb = sp.latex(b)
    if b.is_Symbol or b.is_Integer and b >= 0:
        return lb + "^n"
    return r"\left(" + lb + r"\right)^n"


def _cv_txt_geo_zero(b, q, n):
    """Phrase concrète : q(n,x) b(x)^n -> 0 quand |b|<1."""
    lb = sp.latex(b)
    if sp.simplify(q - 1) == 0:
        return (r"\left|" + lb + r"\right|<1 \text{ : } f_n(x)=" + _cv_pw(b, n) + r" \text{ est une suite géométrique de raison de valeur absolue } <1"
                r"\text{, donc } f_n(x)\longrightarrow 0")
    try:
        dom = _cv_terme_dominant(sp.simplify(q), n) if sp.simplify(q).is_rational_function(n) else None
    except Exception:
        dom = None
    if dom is not None and dom[0] > 0:
        part = (r"\text{ ; le facteur } " + sp.latex(q) + r"\text{ ne croît que comme } n^{" + sp.latex(dom[0])
                + r"}\text{ (polynomial), alors que } \left|" + lb + r"\right|^n \text{ décroît exponentiellement}")
    elif dom is not None and dom[0] < 0:
        part = r"\text{ ; le facteur } " + sp.latex(q) + r"\text{ tend vers } 0 \text{ et } \left|" + lb + r"\right|^n \le 1"
    else:
        part = r"\text{ ; le facteur } " + sp.latex(q) + r"\text{ est borné ou polynomial en } n \text{ et } \left|" + lb + r"\right|^n \text{ décroît exponentiellement}"
    return (r"\left|" + lb + r"\right|<1 \text{, avec } f_n(x)=" + sp.latex(q) + r"\cdot " + _cv_pw(b, n) + part
            + r"\text{ ; donc } f_n(x)\longrightarrow 0 \text{ (croissances comparées)}")


def _cv_limite_ponctuelle(f, n, x, I):
    """Étude de la limite simple de f_n(x) pour x dans I.
    Renvoie une liste de pièces : dict(ens, statut, lim, ligne)."""
    f = sp.simplify(f)
    a, b, _, _ = _cv_extremites(I)
    points = [e for e in (a, b) if e.is_finite]

    # f_n(x) = (facteur borné) * r(n, x) avec r(n, x) -> 0 : une seule pièce suffit
    r0, borne0 = _cv_sep_borne(f, n)
    if borne0:
        st_r, L_r = _cv_lim_n(r0, n)
        if st_r == "cv" and L_r == 0 and not r0.has(x):
            lat = _cv_ens_latex(I)
            return [dict(ens=I, statut="cv", lim=sp.Integer(0),
                         ligne=r"\text{Pour } x\in " + lat + r",\quad " + _cv_txt_borne(f, r0, n))]

    geo = _cv_decomp_geometrique(f, n, x)
    if geo is None:
        points.append(sp.Integer(0))
    if geo is not None:
        base = geo[0]
        for cible in (1, -1):
            try:
                sol = sp.solveset(sp.Eq(base, cible), x, domain=S.Reals)
                if isinstance(sol, sp.FiniteSet):
                    points += list(sol.args)
            except Exception:
                pass
    # zéros (indépendants de n) du dénominateur
    try:
        _, den = sp.fraction(sp.together(f))
        for fac, _m in sp.factor_list(den)[1]:
            if fac.has(x) and not fac.has(n):
                sol = sp.solveset(fac, x, domain=S.Reals)
                if isinstance(sol, sp.FiniteSet):
                    points += list(sol.args)
    except Exception:
        pass

    pieces = []
    for morceau in _cv_decoupe(I, points):
        pieces.append(_cv_piece(f, n, x, morceau, geo))
    return pieces


def _cv_ens_latex(ens):
    if isinstance(ens, sp.FiniteSet):
        return sp.latex(ens.args[0])
    return _latex_intervalle(ens)


def _cv_piece(f, n, x, morceau, geo):
    lat = _cv_ens_latex(morceau)
    # ---- point isolé
    if isinstance(morceau, sp.FiniteSet):
        x0 = morceau.args[0]
        fx0 = _cv_nettoie(sp.simplify(f.subs(x, x0)), n)
        statut, L = _cv_lim_n(fx0, n)
        pre = r"\text{Pour } x=" + lat + r",\quad f_n(" + lat + r")=" + sp.latex(fx0)
        if statut == "cv":
            return dict(ens=morceau, statut="cv", lim=L, ligne=pre + r"\ \longrightarrow\ " + sp.latex(L))
        if statut == "div":
            return dict(ens=morceau, statut="div", lim=None, ligne=pre + r"\ \longrightarrow\ " + sp.latex(L) + r"\ \text{(divergence)}")
        if statut == "osc":
            return dict(ens=morceau, statut="div", lim=None, ligne=pre + r"\ \text{ n'a pas de limite.}")
        return dict(ens=morceau, statut="ind", lim=None, ligne=pre + r"\ \text{ : le calcul symbolique ne conclut pas.}")

    # ---- intervalle ouvert (pas de point remarquable à l'intérieur)
    ech = _cv_echantillons(morceau)
    pre = r"\text{Pour } x\in " + lat + r",\quad "
    if geo is not None:
        b, q = geo
        regimes = set()
        for xv in ech:
            try:
                regimes.add(bool(sp.Abs(b.subs(x, xv)) < 1))
            except Exception:
                regimes.add(None)
        if len(regimes) == 1 and None not in regimes:
            sep = _cv_sep_borne(q, n)
            r, borne = sep
            if r.is_rational_function(n):
                if regimes == {True}:
                    return dict(
                        ens=morceau, statut="cv", lim=sp.Integer(0), geo=(b, q),
                        ligne=pre + _cv_txt_geo_zero(b, q, n),
                    )
                if not borne:
                    dom = _cv_terme_dominant(r, n)
                    if dom is not None:
                        k, c = dom
                        try:
                            zeros = sp.solveset(c, x, domain=morceau) if c.has(x) else (S.EmptySet if c != 0 else morceau)
                        except Exception:
                            zeros = None
                        if zeros == S.EmptySet:
                            return dict(
                                ens=morceau, statut="div", lim=None,
                                ligne=pre + r"\left|" + sp.latex(b) + r"\right|>1 \text{ : la raison est de valeur absolue } >1 \text{ et le facteur } " + sp.latex(q) + r" \text{ ne la compense pas, donc } |f_n(x)|\longrightarrow +\infty \text{ (la suite diverge)}",
                            )

    # ---- cas général : limite générique, puis garde-fou numérique
    L = None
    r, borne = _cv_sep_borne(f, n)
    if borne:
        st_r, L_r = _cv_lim_n(r, n)
        if st_r == "cv" and L_r == 0:
            L = sp.Integer(0)
            ligne = pre + _cv_txt_borne(f, r, n)
    if L is None:
        try:
            L0 = sp.limit(f, n, sp.oo)
            if (not isinstance(L0, sp.Limit)) and (not L0.has(n)) and L0 not in (sp.oo, -sp.oo, sp.zoo, sp.nan):
                if _cv_verif_numerique(f, L0, n, x, morceau):
                    L = sp.simplify(L0)
        except Exception:
            L = None
        if L is not None:
            ligne = pre + r"f_n(x)=" + sp.latex(f) + r"\ \longrightarrow\ " + sp.latex(L)
    if L is not None:
        return dict(ens=morceau, statut="cv", lim=L, ligne=ligne)
    return dict(ens=morceau, statut="ind", lim=None,
                ligne=pre + r"\text{le calcul symbolique ne conclut pas sur la limite simple.}")


def _cv_fusionne(pieces):
    """Regroupe les pièces voisines de même statut et même limite."""
    groupes = []
    for p in pieces:
        for g in groupes:
            if g["statut"] == p["statut"] and (
                (p["lim"] is None and g["lim"] is None) or
                (p["lim"] is not None and g["lim"] is not None and sp.simplify(p["lim"] - g["lim"]) == 0)
            ):
                g["ens"] = sp.Union(g["ens"], p["ens"])
                break
        else:
            groupes.append(dict(ens=p["ens"], statut=p["statut"], lim=p["lim"]))
    return groupes




# -------------------------------------------------------------
# Borne supérieure par étude de variations
# -------------------------------------------------------------
def _cv_composantes(ens):
    """Découpe un ensemble (Interval, FiniteSet, Union) en composantes simples."""
    if isinstance(ens, sp.Union):
        res = []
        for a in ens.args:
            res += _cv_composantes(a)
        return res
    if isinstance(ens, sp.FiniteSet):
        return [sp.FiniteSet(p) for p in ens.args]
    return [ens]


def _cv_garde_sup(h, candidats, n, x, piece):
    """Garde-fou numérique : la grille ne dépasse pas le maximum des candidats."""
    try:
        a, b = piece.start, piece.end
        lo = float(a) if a.is_finite else -60.0
        hi = float(b) if b.is_finite else 60.0
        xs = np.linspace(lo, hi, 4001)
        for N in (30, 31, 60):
            vals = [abs(_cv_val_f(c, n, x, None, N)) for c in candidats]
            vals = [v for v in vals if v is not None]
            if not vals:
                return True
            hn = h.subs(n, N)
            ys = np.abs(_eval_vecteur(x, hn, xs))
            ys = ys[np.isfinite(ys)]
            if ys.size and ys.max() > max(vals) * (1 + 1e-6) + 1e-9:
                return False
        return True
    except Exception:
        return True


def _cv_val_f(expr, n, x, xv, N):
    try:
        e = expr.subs(n, N)
        if xv is not None:
            e = e.subs(x, xv)
        v = sp.N(e, 20)
        if v.is_real:
            return float(v)
    except Exception:
        pass
    return None


def _cv_continue_pour_n(h, n, x, interieur):
    """h est continue sur l'intérieur pour tout n >= 1 (test symbolique, puis test sur des valeurs de n)."""
    try:
        if interieur.is_subset(continuous_domain(h, x, S.Reals)) is True:
            return True
    except Exception:
        pass
    try:
        for N in (1, 2, 3, 40, 41, 80, 200):
            dom = continuous_domain(h.subs(n, N), x, S.Reals)
            if interieur.is_subset(dom) is not True:
                return False
        return True
    except Exception:
        return False


def _cv_nettoie(expr, n):
    """0**n -> 0 (on travaille pour n >= 1)."""
    try:
        return expr.replace(lambda e: e.is_Pow and e.base == 0 and e.exp.has(n), lambda e: sp.Integer(0))
    except Exception:
        return expr


_CV_ETIQ = []


def _cv_ligne_valeurs(nom, n):
    """Ligne expliquant d'où viennent les candidats du maximum (valeurs aux points critiques et aux bornes)."""
    if not _CV_ETIQ:
        return None
    morceaux = []
    for pt, val in _CV_ETIQ:
        morceaux.append(r"\text{en } " + pt + r"\text{ : } " + sp.latex(val))
    return (r"\text{Valeurs de } |" + nom + r"| \text{ aux points à comparer (point critique, bornes) : } " + r";\ ".join(morceaux)
            + r"\text{. Le sup est la plus grande de ces valeurs.}")


def _cv_sup_candidats(h, n, x, piece):
    """Candidats pour sup_{x in piece} |h(x)| (valeurs aux points critiques et aux bornes).
    Renvoie (liste de candidats, points critiques) ou None si l'étude échoue.
    Chaque candidat est une valeur réellement prise (ou une limite) : c'est un minorant du sup.
    Pour en faire un majorant, l'ensemble des points critiques doit être complet (garde-fou numérique)."""
    npos = sp.Symbol("n", integer=True, positive=True)
    del _CV_ETIQ[:]
    h = sp.simplify(h.subs(n, npos))
    retour = lambda e: _cv_nettoie(sp.simplify(e.subs(npos, n)), n) if hasattr(e, "subs") else e
    if isinstance(piece, sp.FiniteSet):
        return [retour(sp.Abs(h.subs(x, piece.args[0])))], []
    a, b = piece.start, piece.end
    interieur = sp.Interval.open(a, b)
    if not _cv_continue_pour_n(h, npos, x, interieur):
        return None
    hp = sp.simplify(sp.diff(h, x))
    cands, crits = [], []
    if hp == 0:
        return [retour(sp.Abs(h))], []
    try:
        sols = sp.solve(hp, x)
    except Exception:
        return None
    if not isinstance(sols, list):
        return None
    for c in sols:
        if c.has(x) or c.is_real is False:
            return None
        ok = True
        for N in (40, 41, 80):
            try:
                cn = c.subs(npos, N) if c.has(npos) else c
                if interieur.contains(cn) != S.true:
                    ok = False
            except Exception:
                ok = False
        if ok:
            crits.append(retour(c))
            cands.append(sp.Abs(sp.simplify(h.subs(x, c))))
            _CV_ETIQ.append(("x=" + sp.latex(retour(c)), retour(sp.Abs(sp.simplify(h.subs(x, c))))))
    for e, ouvert, sens in ((a, piece.left_open, "+"), (b, piece.right_open, "-")):
        try:
            if e.is_finite and not ouvert:
                val = h.subs(x, e)
            elif e.is_finite:
                val = sp.limit(h, x, e, sens)
            else:
                val = sp.limit(h, x, e)
            if isinstance(val, sp.Limit) or val is sp.nan:
                return None
            cands.append(sp.oo if val.is_infinite else sp.Abs(sp.simplify(val)))
            _CV_ETIQ.append((("x=" + sp.latex(e)) if (e.is_finite and not ouvert) else ("x\\to " + sp.latex(e) + ("^{+}" if sens == "+" else "^{-}") if e.is_finite else "x\\to " + sp.latex(e)),
                             sp.oo if val.is_infinite else retour(sp.Abs(sp.simplify(val)))))
        except Exception:
            return None
    finals = []
    for c in cands:
        c2 = c if c == sp.oo else retour(c)
        if c2 not in finals:
            finals.append(c2)
    if not any(c == sp.oo for c in finals):
        if not _cv_garde_sup(h.subs(npos, n), finals, n, x, piece):
            return None
    return finals, crits


def _cv_verdict_candidats(cands, n):
    """'zero' si tous les candidats tendent vers 0, 'non' si l'un ne tend pas vers 0, None sinon."""
    inconnu = False
    for c in cands:
        if c == sp.oo:
            return "non", c
        st_, L = _cv_lim_n(c, n)
        if st_ == "cv" and L == 0:
            continue
        if st_ in ("cv", "div"):
            return "non", c
        inconnu = True
    return (None if inconnu else "zero"), None


# -------------------------------------------------------------
# Suites de fonctions : convergence uniforme
# -------------------------------------------------------------
def _cv_continue_sur(expr, x, I, n=None):
    try:
        dom = continuous_domain(expr, x, S.Reals)
        if I.is_subset(dom) is True:
            return True
    except Exception:
        pass
    if n is not None and expr.has(n):
        try:
            for N in (1, 2, 3, 40, 41, 80, 200):
                if I.is_subset(continuous_domain(expr.subs(n, N), x, S.Reals)) is not True:
                    return False
            return True
        except Exception:
            return False
    return False


def _cv_cond_latex(ens):
    if isinstance(ens, sp.FiniteSet) and len(ens.args) == 1:
        return r"x=" + sp.latex(ens.args[0])
    return r"x\in " + _cv_ens_latex(ens)


def _cv_expr_limite(groupes_cv):
    """Écriture LaTeX de f(x) à partir des groupes convergents."""
    if len(groupes_cv) == 1:
        return sp.latex(groupes_cv[0]["lim"])
    lignes = [sp.latex(g["lim"]) + r" & " + _cv_cond_latex(g["ens"]) for g in groupes_cv]
    return r"\begin{cases}" + r"\\".join(lignes) + r"\end{cases}"


def _cv_discontinuite_limite(pieces, x):
    """Cherche un point où la limite simple f est discontinue. Renvoie (x0, gauche, droite, valeur) ou None."""
    pts = [p for p in pieces if isinstance(p["ens"], sp.FiniteSet) and p["statut"] == "cv"]
    for p in pts:
        x0 = p["ens"].args[0]
        v0 = p["lim"]
        for q in pieces:
            if q is p or q["statut"] != "cv" or isinstance(q["ens"], sp.FiniteSet):
                continue
            a, b = q["ens"].start, q["ens"].end
            sens = None
            if a == x0:
                sens = "+"
            elif b == x0:
                sens = "-"
            if sens is None:
                continue
            try:
                lim = sp.limit(q["lim"], x, x0, sens) if q["lim"].has(x) else q["lim"]
                if sp.simplify(lim - v0) != 0:
                    return x0, lim, v0
            except Exception:
                continue
    return None


_CV_SUITES_TEST = [
    "1/n", "1/n**2", "1/sqrt(n)", "pi/(2*n)", "pi/n", "1-1/n", "1-1/n**2", "1-1/sqrt(n)",
    "n/(n+1)", "n", "2*n", "log(n)", "1/log(n)",
]


def _cv_suites_test(f_n_minus_f, pieces, n, x, I):
    """Cherche x_n dans I avec |f_n(x_n)-f(x_n)| qui ne tend pas vers 0.
    f_n_minus_f(piece) donne h sur la pièce. Renvoie (x_n, expression de l'écart, limite) ou None."""
    a, b, _, _ = _cv_extremites(I)
    cand = [sp.sympify(c, locals={"n": n}) for c in _CV_SUITES_TEST]
    t_list = [sp.sympify(c, locals={"n": n}) for c in ("1/n", "1/n**2", "pi/(2*n)")]
    for e in (a, b):
        if e.is_finite:
            cand += [e + t for t in t_list] + [e - t for t in t_list]
    for xn in cand:
        if xn.has(x):
            continue
        piece = None
        ok = True
        for N in (40, 41, 80, 81):
            try:
                v = xn.subs(n, N)
                if I.contains(v) != S.true:
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if not ok:
            continue
        try:
            v80 = xn.subs(n, 80)
            for p in pieces:
                if p["statut"] == "cv" and p["ens"].contains(v80) == S.true:
                    piece = p
                    break
            if piece is None:
                continue
            h = f_n_minus_f(piece)
            ecart = sp.simplify(sp.Abs(sp.simplify(h.subs(x, xn))))
            st_, L = _cv_lim_n(ecart, n)
            if (st_ == "cv" and L != 0) or st_ == "div":
                return xn, ecart, L
        except Exception:
            continue
    return None




def _cv_fonction_limite_plot(groupes_cv, x):
    """Expression utilisable pour le tracé (Piecewise) de la limite simple."""
    if not groupes_cv:
        return None
    if len(groupes_cv) == 1 and not isinstance(groupes_cv[0]["ens"], sp.FiniteSet):
        return groupes_cv[0]["lim"]
    try:
        morceaux = []
        for g in groupes_cv:
            for comp in _cv_composantes(g["ens"]):
                morceaux.append((g["lim"], comp.as_relational(x)))
        return sp.Piecewise(*morceaux)
    except Exception:
        return None


def _cv_suite_analyse(f_n, n, x, I, n_min, profondeur=1):
    """Étude complète d'une suite de fonctions. Renvoie un dictionnaire de lignes LaTeX et de verdicts."""
    res = dict(simple=[], unif=[], cons=[], v_simple=None, v_unif=None, groupes=[], pieces=[])
    latI = _latex_intervalle(I)
    pieces = _cv_limite_ponctuelle(f_n, n, x, I)
    res["pieces"] = pieces
    res["simple"] = [p["ligne"] for p in pieces]
    groupes = _cv_fusionne(pieces)
    res["groupes"] = groupes
    cv = [g for g in groupes if g["statut"] == "cv"]
    pas_cv = [g for g in groupes if g["statut"] != "cv"]

    # ---------------- convergence simple : conclusion
    if not pas_cv:
        res["v_simple"] = True
        res["simple"].append(
            r"\text{Ainsi } (f_n) \text{ converge simplement sur } " + latI
            + r" \text{ vers } f(x)=" + _cv_expr_limite(cv)
        )
    else:
        if any(g["statut"] == "ind" for g in pas_cv):
            res["v_simple"] = None
            res["simple"].append(
                r"\text{Le calcul symbolique ne permet pas de conclure sur la convergence simple sur } "
                + sp.latex(sp.Union(*[g["ens"] for g in pas_cv if g["statut"] == "ind"]))
                + r"."
            )
        else:
            res["v_simple"] = False
        if cv:
            E = sp.Union(*[g["ens"] for g in cv])
            res["simple"].append(
                r"\text{Ainsi } (f_n) \text{ converge simplement sur } " + sp.latex(E)
                + r" \text{ vers } f(x)=" + _cv_expr_limite(cv)
            )
        if res["v_simple"] is False:
            res["simple"].append(
                r"\text{La suite } (f_n) \text{ diverge pour } x\in "
                + sp.latex(sp.Union(*[g["ens"] for g in pas_cv]))
                + r"\text{ : elle ne converge pas simplement sur } " + latI + r"."
            )

    # ---------------- convergence uniforme
    if res["v_simple"] is False:
        res["v_unif"] = False
        res["unif"].append(
            r"\text{La convergence uniforme sur } " + latI
            + r" \text{ entraînerait la convergence simple sur } " + latI
            + r"\text{ : } (f_n) \text{ ne converge donc pas uniformément sur } " + latI + r"."
        )
        return res
    if res["v_simple"] is None:
        return res

    cont_n = _cv_continue_sur(f_n, x, I, n)

    # (1) théorème de continuité
    if cont_n:
        d = _cv_discontinuite_limite(pieces, x)
        if d is not None:
            x0, lim, v0 = d
            res["unif"].append(
                r"\text{Chaque } f_n \text{ est continue sur } " + latI
                + r"\text{. Or } \lim_{x\to " + sp.latex(x0) + r"} f(x)=" + sp.latex(lim)
                + r"\ne " + sp.latex(v0) + r"=f(" + sp.latex(x0) + r"),\ \text{donc } f \text{ n'est pas continue en } " + sp.latex(x0) + r"."
            )
            res["unif"].append(
                r"\text{Une limite uniforme de fonctions continues est continue : } (f_n) \text{ ne converge pas uniformément sur } " + latI + r"."
            )
            res["v_unif"] = False
            return res

    # (2) étude de variations de h_n = f_n - f (ou majoration si h contient sin/cos)
    lignes_var, tous = [], []
    echec = False
    majorant = False
    for g in cv:
        h = sp.simplify(f_n - g["lim"])
        for comp in _cv_composantes(g["ens"]):
            lat = _cv_ens_latex(comp)
            trig = h.has(sp.sin) or h.has(sp.cos) or h.has(sp.tan)
            if trig and not isinstance(comp, sp.FiniteSet):
                r_h, borne = _cv_sep_borne(h, n)
                if not borne or r_h.has(sp.sin) or r_h.has(sp.cos) or r_h.has(sp.tan):
                    echec = True
                    break
                majorant = True
                lignes_var.append(
                    r"\text{Pour tout } x\in " + lat + r",\quad |f_n(x)-f(x)|=\left|" + sp.latex(h)
                    + r"\right|\le \left|" + sp.latex(r_h) + r"\right|\quad \text{car les facteurs trigonométriques de } " + sp.latex(sp.simplify(h / r_h)) + r" \text{ sont bornés par 1 en valeur absolue.}"
                )
                if r_h.has(x):
                    r = _cv_sup_candidats(r_h, n, x, comp)
                    if r is None:
                        echec = True
                        break
                    tous += r[0]
                else:
                    tous.append(sp.simplify(sp.Abs(r_h)))
                continue
            r = _cv_sup_candidats(h, n, x, comp)
            if r is None:
                echec = True
                break
            cands, crits = r
            if isinstance(comp, sp.FiniteSet):
                lignes_var.append(r"\text{En } x=" + lat + r",\quad |f_n(x)-f(x)|=" + sp.latex(cands[0]))
            else:
                lignes_var.append(
                    r"\text{Sur } " + lat + r",\quad h_n(x)=f_n(x)-f(x)=" + sp.latex(h)
                )
                hp = sp.simplify(sp.diff(h, x))
                lignes_var.append(r"h_n'(x)=" + sp.latex(hp))
                if crits:
                    lignes_var.append(
                        r"\text{Cette dérivée s'annule en } x=" + r",\ ".join(sp.latex(c) for c in crits)
                        + r"\text{ (point critique situé dans } " + lat + r"\text{ pour } n \text{ assez grand).}"
                    )
                else:
                    lignes_var.append(r"\text{Cette dérivée ne s'annule pas à l'intérieur de } " + lat + r"\text{ : } h_n \text{ y est monotone, son maximum en valeur absolue est atteint aux bornes.}")
                lv = _cv_ligne_valeurs("h_n(x)", n)
                if lv:
                    lignes_var.append(lv)
            tous += cands
        if echec:
            break
    if not echec and tous:
        utiles = [c for c in tous if c != 0] or [sp.Integer(0)]
        res["unif"] += lignes_var
        signe = r"\le" if majorant else r"="
        if len(utiles) == 1:
            res["unif"].append(r"M_n=\sup_{x\in " + latI + r"}|f_n(x)-f(x)|" + signe + sp.latex(utiles[0]))
        else:
            res["unif"].append(
                r"M_n=\sup_{x\in " + latI + r"}|f_n(x)-f(x)|" + signe + r"\max\left(" + r",\ ".join(sp.latex(c) for c in utiles) + r"\right)"
            )
        verdict, mauvais = _cv_verdict_candidats(utiles, n)
        if verdict == "zero":
            res["unif"].append(
                r"\text{Comme } M_n\longrightarrow 0,\ (f_n) \text{ converge uniformément sur } " + latI + r" \text{ vers } f."
            )
            res["v_unif"] = True
        elif verdict == "non" and not majorant:
            res["unif"].append(
                r"\text{Or } M_n\ge " + sp.latex(mauvais) + r" \text{ ne tend pas vers } 0\text{ : } (f_n) \text{ ne converge pas uniformément sur } " + latI + r"."
            )
            res["v_unif"] = False
        elif majorant:
            res["unif"] = [l for l in res["unif"] if l not in lignes_var] if False else res["unif"]

    # (3) suites test
    if res["v_unif"] is None:
        def h_piece(p):
            return sp.simplify(f_n - p["lim"])
        t = _cv_suites_test(h_piece, pieces, n, x, I)
        if t is not None:
            xn, ecart, L = t
            res["unif"].append(
                r"\text{Posons } x_n=" + sp.latex(xn) + r"\in " + latI
                + r"\text{. Alors } |f_n(x_n)-f(x_n)|=" + sp.latex(ecart)
                + r"\longrightarrow " + sp.latex(L) + r"\ne 0\text{.}"
            )
            res["unif"].append(
                r"\text{Donc } \sup_{x\in " + latI + r"}|f_n(x)-f(x)|\not\to 0\text{ : } (f_n) \text{ ne converge pas uniformément sur } " + latI + r"."
            )
            res["v_unif"] = False
    xn_ = _cv_lignes_x_sur_n(f_n, n, x, I, pieces) if res["v_unif"] is not True else None
    if res["v_unif"] is None and xn_ is not None and I.is_subset(sp.Interval(-sp.oo, sp.oo)) and (
        I != sp.S.Reals and I.start.is_finite and I.end.is_finite
    ):
        res["unif"] += [xn_[0], xn_[1],
                        r"\text{Donc } (f_n) \text{ converge uniformément sur } " + latI + r" \text{ (intervalle borné).}"]
        res["v_unif"] = True
    if res["v_unif"] is None:
        res["unif"].append(
            r"\text{Le calcul symbolique ne permet pas de conclure sur la convergence uniforme sur } " + latI + r"."
        )
        if xn_ is not None:
            res["unif"] += [xn_[0], xn_[1], r"\text{Donc } (f_n) \text{ converge uniformément sur tout segment de } \mathbb{R}\text{.}"]
        res["unif"] += _cv_lignes_segments_suite(pieces, n, x)
        return res

    if res["v_unif"] is False:
        res["unif"] += _cv_lignes_segments_suite(pieces, n, x)
        if xn_ is not None:
            res["unif"] += [xn_[0], xn_[1], r"\text{Donc } (f_n) \text{ converge uniformément sur tout segment inclus dans } " + latI + r"\text{.}"]

    # ---------------- conséquences (si convergence uniforme)
    if res["v_unif"] is True:
        if cont_n:
            res["cons"].append(
                r"\text{Chaque } f_n \text{ est continue sur } " + latI
                + r" \text{ et } (f_n) \text{ converge uniformément : la limite } f \text{ est continue sur } " + latI + r"."
            )
            if _cv_est_segment(I):
                a_, b_ = I.start, I.end
                texte = (r"\text{Intégration sur un segment : } \lim_{n\to+\infty}\int_{" + sp.latex(a_) + r"}^{" + sp.latex(b_)
                         + r"} f_n(x)\,dx=\int_{" + sp.latex(a_) + r"}^{" + sp.latex(b_) + r"} f(x)\,dx")
                try:
                    if len(cv) == 1 and len(_cv_composantes(cv[0]["ens"])) <= 2:
                        val = sp.simplify(sp.integrate(cv[0]["lim"], (x, a_, b_)))
                        texte += r"=" + sp.latex(val)
                except Exception:
                    pass
                res["cons"].append(texte)
        if profondeur > 0:
            try:
                fp = sp.simplify(sp.diff(f_n, x))
                if _cv_continue_sur(fp, x, I, n) and _cv_continue_sur(f_n, x, I, n):
                    sous = _cv_suite_analyse(fp, n, x, I, n_min, profondeur=0)
                    if sous["v_unif"] is True and sous["v_simple"] is True and len(sous["groupes"]) >= 1:
                        g_lat = _cv_expr_limite([g for g in sous["groupes"] if g["statut"] == "cv"])
                        res["cons"].append(
                            r"\text{Dérivation : } f_n'(x)=" + sp.latex(fp)
                            + r"\text{ converge uniformément sur } " + latI + r" \text{ vers } g(x)=" + g_lat
                            + r"\text{, donc } f \text{ est de classe } C^1 \text{ et } f'=g."
                        )
            except Exception:
                pass
    return res


def _cv_segments_geo(pieces, n, x, borne_limite_nulle=True):
    """Intervalles J où |b|<1 (décomposition q*b^n, limite nulle) : renvoie la liste des (J, b, q)."""
    res = []
    for p in pieces:
        geo = p.get("geo")
        if geo is None or p["statut"] != "cv" or isinstance(p["ens"], sp.FiniteSet):
            continue
        b, q = geo
        J = p["ens"]
        ok = True
        for N in (1, 5, 40):
            try:
                if not _cv_continue_sur(q.subs(n, N), x, J) or not _cv_continue_sur(b, x, J):
                    ok = False
            except Exception:
                ok = False
        if ok:
            res.append((J, b, q))
    return res


def _cv_lignes_x_sur_n(f_n, n, x, I, pieces):
    """f_n(x)=phi(x/n) avec phi de classe C^1 : renvoie (lignes_segment, ligne_bornee) ou None."""
    try:
        t = sp.Symbol("t", real=True)
        npos = sp.Symbol("n", integer=True, positive=True)
        g = sp.simplify(f_n.subs(n, npos).subs(x, npos * t))
        if g.has(npos) or not g.has(t):
            return None
        L = g.subs(t, 0)
        if not L.is_number or L.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            return None
        gp = sp.simplify(sp.diff(g, t))
        if continuous_domain(gp, t, sp.Interval(-1, 1)) != sp.Interval(-1, 1):
            return None
        if any(p["statut"] != "cv" or sp.simplify(p["lim"] - L) != 0 for p in pieces):
            return None
        intro = (r"\text{On écrit } f_n(x)=\varphi\left(\frac{x}{n}\right) \text{ avec } \varphi(t)=" + sp.latex(g)
                 + r"\text{, de classe } C^1 \text{ sur } [-1,1]\text{ ; } f(x)=\varphi(0)=" + sp.latex(L) + r"\text{.}")
        ineg = (r"\text{Soit } [\alpha,\beta] \text{ un segment et } R=\max(|\alpha|,|\beta|)\text{. Pour } n\ge R \text{ et } x\in[\alpha,\beta]\text{, on a } \left|\frac{x}{n}\right|\le 1\text{. "
                r"Comme } \varphi \text{ est de classe } C^1 \text{ sur } [-1,1]\text{, } C=\sup_{[-1,1]}|\varphi'|<+\infty \text{ et l'inégalité des accroissements finis donne :}"
                r"\quad |f_n(x)-f(x)|=\left|\varphi\left(\frac{x}{n}\right)-\varphi(0)\right|\le C\left|\frac{x}{n}-0\right|=C\,\frac{|x|}{n}\le \frac{C\,R}{n}\underset{n\to+\infty}{\longrightarrow} 0\text{.}")
        return intro, ineg
    except Exception:
        return None


def _cv_lignes_segments_suite(pieces, n, x):
    lignes = []
    for J, b, q in _cv_segments_geo(pieces, n, x):
        lat = _cv_ens_latex(J)
        lb = sp.latex(b)
        q = sp.simplify(q)
        triv = sp.simplify(q - 1) == 0
        lignes.append(
            r"\text{Soit } [\alpha,\beta]\subset " + lat + r"\text{ un segment quelconque. La fonction } x\mapsto \left|" + lb
            + r"\right| \text{ est continue sur ce segment fermé, donc atteint son maximum } \rho=\max_{x\in[\alpha,\beta]}\left|" + lb
            + r"\right|\text{ ; ce maximum est atteint en un point de } [\alpha,\beta]\subset " + lat + r"\text{, donc } \rho<1\text{.}"
        )
        if triv:
            lignes.append(
                r"\text{Pour tout } x\in[\alpha,\beta]\text{ et tout } n:\quad |f_n(x)|=\left|" + lb + r"\right|^n\le \rho^{n}"
                r"\underset{n\to+\infty}{\longrightarrow} 0 \text{ (suite géométrique de raison } \rho<1\text{).}"
            )
        else:
            try:
                dom = _cv_terme_dominant(q, n) if q.is_rational_function(n) else None
            except Exception:
                dom = None
            if not q.has(x):
                rq = sp.simplify(q.subs(n, n + 1) / q)
                lignes.append(
                    r"\text{Pour tout } x\in[\alpha,\beta]:\quad |f_n(x)|=|" + sp.latex(q) + r"|\cdot\left|" + lb + r"\right|^n\le " + sp.latex(q) + r"\,\rho^{n}"
                    r"\text{ ; or } \frac{" + sp.latex(q.subs(n, n + 1)) + r"\,\rho^{n+1}}{" + sp.latex(q) + r"\,\rho^{n}}=" + sp.latex(rq) + r"\,\rho\to\rho<1"
                    r"\text{, donc } " + sp.latex(q) + r"\,\rho^n\to 0 \text{ (terme général d'une série convergente par d'Alembert).}"
                )
                lignes.append(
                    r"\text{La borne } " + sp.latex(q) + r"\,\rho^n \text{ ne dépend pas de } x\text{ : } (f_n) \text{ converge uniformément sur tout segment inclus dans } " + lat + r"."
                )
                continue
            if q.has(x) or dom is None:
                borne_q = r"|" + sp.latex(q) + r"|\le C\,n^{" + (sp.latex(dom[0]) if dom else r"k") + r"}"
                expl = (r"\text{ où } C \text{ est une constante qui ne dépend ni de } n \text{ ni de } x \text{ (maximum de la partie en } x \text{ de } "
                        + sp.latex(q) + r" \text{ sur } [\alpha,\beta])")
            else:
                borne_q = r"|" + sp.latex(q) + r"|\le C\,n^{" + sp.latex(dom[0]) + r"}"
                expl = r"\text{ avec } C=" + sp.latex(sp.Abs(dom[1])) + r"\text{ (pour } n\ge 1)" if dom[0] >= 0 else r"\text{ (pour } n\ge 1)"
            kk = sp.latex(dom[0]) if dom else r"k"
            lignes.append(
                r"\text{Pour tout } x\in[\alpha,\beta]:\quad |f_n(x)|=|" + sp.latex(q) + r"|\cdot\left|" + lb + r"\right|^n\le |" + sp.latex(q) + r"|\,\rho^{n}"
                r"\text{ et } " + borne_q + expl + r"\text{, donc } |f_n(x)|\le C\,n^{" + kk + r"}\rho^{n}\underset{n\to+\infty}{\longrightarrow} 0"
                r"\text{ car } n^{" + kk + r"}\rho^n\to 0 \text{ pour } \rho<1 \text{ (croissances comparées).}"
            )
        lignes.append(
            r"\text{La borne } \rho^n \text{ ne dépend pas de } x\text{ : } (f_n) \text{ converge uniformément sur tout segment inclus dans } " + lat + r"."
        )
    return lignes




# -------------------------------------------------------------
# Séries de fonctions
# -------------------------------------------------------------
def _cv_signe_sur(expr, x, K, n):
    """'pos' si expr >= 0 sur K (pour n >= 1), 'neg' si expr <= 0, sinon None."""
    try:
        npos = sp.Symbol("n", integer=True, positive=True)
        a, b = K.start, K.end
        if a.is_finite and a >= 0:
            xs = sp.Symbol("xs", nonnegative=True)
        elif b.is_finite and b <= 0:
            xs = sp.Symbol("xs", nonpositive=True)
        else:
            xs = sp.Symbol("xs", real=True)
        e = sp.simplify(expr.subs({n: npos, x: xs}))
        if e.is_nonnegative:
            return "pos"
        if e.is_nonpositive:
            return "neg"
    except Exception:
        pass
    return None


def _cv_classe(res, lignes):
    """Verdict d'une étude numérique critere(...) : True, False ou None."""
    if res is True:
        return True
    if res is False:
        return False
    texte = " ".join(lignes)
    if "diverge" in texte or "divergente" in texte:
        return False
    return None


class _CvRep(list):
    """Lignes renvoyées quand une série numérique a déjà été étudiée plus haut (évite la redondance)."""


_CV_MEMO = {}


def _cv_serie_numerique(u0, n, n_min, memo=False):
    """Étude d'une série numérique avec les critères existants (critere). Renvoie (verdict, lignes)."""
    u0 = _cv_nettoie(sp.simplify(u0), n)
    cle = (sp.srepr(u0), n_min)
    if memo and cle in _CV_MEMO:
        v0 = _CV_MEMO[cle]
        mot = {True: "converge", False: "diverge"}.get(v0)
        if mot:
            return v0, _CvRep([r"\text{la série } \sum " + sp.latex(u0) + r"\text{ a déjà été étudiée ci-dessus : elle " + mot + r".}"])
    v, lg = _cv_serie_numerique_calc(u0, n, n_min)
    if memo:
        _CV_MEMO[cle] = v
    return v, lg


def _cv_serie_numerique_calc(u0, n, n_min):
    if u0 == 0:
        return True, [r"\text{Tous les termes sont nuls : la série converge.}"]
    # règle rapide : équivalent en C/n^p (séries de Riemann), sans passer par critere
    try:
        npos = sp.Symbol("n", integer=True, positive=True)
        w = u0.subs(n, npos)
        if w.is_positive is not False and not w.has(sp.Abs):
            p = sp.limit(-sp.log(w) / sp.log(npos), npos, sp.oo)
            if p.is_number and p.is_finite:
                C = sp.limit(w * npos**p, npos, sp.oo)
                if C.is_number and C.is_finite and C > 0:
                    lg = [r"u_n \sim " + ("" if C == 1 else sp.latex(C) + r"\,") + (r"\frac{1}{n}" if p == 1 else (r"\frac{1}{n^{" + sp.latex(p) + r"}}" if p > 0 else (r"n^{" + sp.latex(-p) + r"}" if p < 0 else "1"))) + r"\quad(n\to+\infty)"]
                    if p <= 0:
                        lg.append(r"\text{Le terme général ne tend pas vers } 0 \text{ : la série diverge grossièrement.}")
                        return False, lg
                    if p == 1:
                        lg.append(r"\text{Série harmonique (à une constante près) : la série diverge.}")
                        return False, lg
                    if p > 1:
                        lg.append(r"\text{Série de Riemann d'exposant } " + sp.latex(p) + r" > 1 \text{ : la série converge.}")
                        return True, lg
                    lg.append(r"\text{Série de Riemann d'exposant } " + sp.latex(p) + r" < 1 \text{ : la série diverge.}")
                    return False, lg
    except Exception:
        pass
    res, lignes = _cv_capture(critere, u0, n, n_min)
    return _cv_classe(res, lignes), lignes


def _cv_leibniz_serie(u_n, n, x, K):
    """u_n = (-1)^n a_n(x) avec a_n >= 0 décroissante en n. Renvoie (a_n, lignes) ou None."""
    try:
        u_n = sp.expand_power_base(u_n, force=True)
        alt = [p for p in u_n.atoms(sp.Pow) if p.base == -1 and p.exp.has(n)]
        if len(alt) != 1:
            return None
        a = sp.simplify(u_n / alt[0])
        if a.has(n) is False:
            return None
        signe = _cv_signe_sur(a, x, K, n)
        if signe == "neg":
            a = sp.simplify(-a)
        elif signe != "pos":
            return None
        d = sp.simplify(a.subs(n, n + 1) - a)
        if _cv_signe_sur(d, x, K, n) != "neg":
            # repli : a_{n+1}(x)/a_n(x) <= 1 sur K
            rapport = sp.simplify(a.subs(n, n + 1) / a)
            r = _cv_sup_candidats(rapport, n, x, K) if isinstance(K, sp.Interval) else None
            if r is None:
                return None
            npos = sp.Symbol("n", integer=True, positive=True)
            for c in r[0]:
                if c == sp.oo or sp.simplify(c.subs(n, npos) - 1).is_nonpositive is not True:
                    return None
        return a, d
    except Exception:
        return None


def _cv_ratio_piece(u_n, n, x, comp):
    """Règle de d'Alembert sur un intervalle : renvoie ('cv'|'div'|None, rho)."""
    try:
        r = sp.simplify(sp.Abs(sp.simplify(u_n.subs(n, n + 1) / u_n)))
        st_, L = _cv_lim_n(r, n)
        if st_ != "cv":
            return None, None
        if not L.has(x):
            if L < 1:
                return "cv", L
            if L > 1:
                return "div", L
            return None, None
        if sp.solveset(sp.Ge(L, 1), x, domain=comp) == S.EmptySet:
            return "cv", L
        if sp.solveset(sp.Le(L, 1), x, domain=comp) == S.EmptySet:
            return "div", L
    except Exception:
        pass
    return None, None


def _cv_motif_serie(c, n):
    """Courte justification concrète de la convergence de la série numérique de terme général c(n) >= 0."""
    lc = sp.latex(c)
    try:
        if c.is_rational_function(n):
            num, den = sp.fraction(sp.together(c))
            p = sp.Poly(den, n).degree() - sp.Poly(num, n).degree()
            C = sp.limit(c * n**p, n, sp.oo)
            if p > 1 and not C.has(n) and C not in (0, sp.oo, -sp.oo):
                return (r"\text{ ; } " + lc + r"\sim \frac{" + sp.latex(C) + r"}{n^{" + sp.latex(p) + r"}} \text{ : série de Riemann d'exposant } "
                        + sp.latex(p) + r">1\text{, donc } \sum " + lc + r"\text{ converge}")
    except Exception:
        pass
    try:
        rr = sp.simplify(c.subs(n, n + 1) / c)
        st_, L = _cv_lim_n(rr, n)
        if st_ == "cv" and L.is_number and L < 1:
            return (r"\text{ ; } \frac{M_{n+1}}{M_n}=" + sp.latex(rr) + r"\longrightarrow " + sp.latex(L)
                    + r"<1 \text{ : par d'Alembert, } \sum M_n \text{ converge}")
    except Exception:
        pass
    return r"\text{ ; la série numérique } \sum M_n \text{ converge}"


def _cv_serie_pieces(u_n, n, x, I, n_min):
    """Nature de la série en chaque x de I. Renvoie une liste de dict(ens, nature, lignes, geo)."""
    pieces = _cv_limite_ponctuelle(u_n, n, x, I)
    sortie = []
    for p in pieces:
        ens = p["ens"]
        lat = _cv_ens_latex(ens)
        pre = r"\text{Pour } " + (r"x=" + lat if isinstance(ens, sp.FiniteSet) else r"x\in " + lat) + r",\quad "
        # la condition nécessaire u_n(x) -> 0
        if p["statut"] in ("div",) or (p["statut"] == "cv" and p["lim"] != 0):
            base_l = p["ligne"].replace("f_n", "u_n")
            for fin in (r"\ \text{(divergence)}", r"\ \text{ n'a pas de limite.}"):
                if base_l.endswith(fin):
                    base_l = base_l[:-len(fin)]
            sortie.append(dict(ens=ens, nature="div", lignes=[
                base_l + r"\ \text{ ne tend pas vers } 0\text{ : la série } \sum u_n \text{ diverge grossièrement.}"], geo=None))
            continue
        if p["statut"] == "ind":
            sortie.append(dict(ens=ens, nature="ind", lignes=[p["ligne"].replace("f_n", "u_n")], geo=None))
            continue
        # u_n(x) -> 0
        if isinstance(ens, sp.FiniteSet):
            u0 = u_n.subs(x, ens.args[0])
            v, lg = _cv_serie_numerique(u0, n, n_min, memo=True)
            nat = "cv" if v is True else ("div" if v is False else "ind")
            if isinstance(lg, _CvRep):
                sortie.append(dict(ens=ens, nature=nat, lignes=[r"\text{Pour } x=" + lat + r",\quad " + lg[0]], geo=None))
                continue
            tete = r"\text{Pour } x=" + lat + r",\quad \text{étude de la série numérique } \sum " + sp.latex(_cv_nettoie(sp.simplify(u0), n))
            sortie.append(dict(ens=ens, nature=nat, lignes=[tete] + lg, geo=None))
            continue
        if p.get("geo") is not None:
            b, q = p["geo"]
            try:
                npos_ = sp.Symbol("n", integer=True, positive=True)
                rq = sp.simplify(sp.Abs(sp.simplify(q.subs(n, n + 1) / q)).subs(n, npos_)).subs(npos_, n)
                st_, Lq = _cv_lim_n(rq, n)
                rr = sp.Mul(rq, sp.Abs(b), evaluate=False) if rq != 1 else sp.Abs(b)
                Lr = sp.Abs(b) if (st_ == "cv" and Lq == 1) else None
                if Lr is None:
                    st_ = None
            except Exception:
                rr, st_, Lr = None, None, None
            if rr is not None and st_ == "cv":
                lg_ = (pre + r"\left|\frac{u_{n+1}(x)}{u_n(x)}\right|=" + sp.latex(rr) + r"\underset{n\to+\infty}{\longrightarrow} " + sp.latex(Lr)
                       + r"<1 \text{ : d'Alembert, } \sum u_n(x) \text{ converge absolument.}")
            else:
                lg_ = (pre + r"|u_n(x)|=|" + sp.latex(q) + r"|\cdot\left|" + sp.latex(b) + r"\right|^n \text{ avec } \left|" + sp.latex(b)
                       + r"\right|<1 \text{ : comparaison à une série géométrique, } \sum u_n(x) \text{ converge absolument.}")
            sortie.append(dict(ens=ens, nature="cv_abs", geo=p["geo"], lignes=[lg_]))
            continue
        rho_nat, rho = _cv_ratio_piece(u_n, n, x, ens)
        if rho_nat == "cv":
            sortie.append(dict(ens=ens, nature="cv_abs", geo=None, lignes=[
                pre + r"\lim_{n\to+\infty}\left|\frac{u_{n+1}(x)}{u_n(x)}\right|=" + sp.latex(rho) + r"<1\text{ : d'Alembert, } \sum u_n(x) \text{ converge absolument.}"]))
            continue
        if rho_nat == "div":
            sortie.append(dict(ens=ens, nature="div", geo=None, lignes=[
                pre + r"\lim_{n\to+\infty}\left|\frac{u_{n+1}(x)}{u_n(x)}\right|=" + sp.latex(rho) + r">1\text{ : d'Alembert, } \sum u_n(x) \text{ diverge.}"]))
            continue
        # domination par une série convergente indépendante de x
        r, borne = _cv_sep_borne(u_n, n)
        if borne and not r.has(x):
            v, lg = _cv_serie_numerique(sp.Abs(r), n, n_min)
            if v is True:
                sortie.append(dict(ens=ens, nature="cv_abs", geo=None, lignes=[
                    pre + r"|u_n(x)|\le \left|" + sp.latex(r) + r"\right|\text{ et } \sum \left|" + sp.latex(r) + r"\right| \text{ converge : } \sum u_n(x) \text{ converge absolument (comparaison).}"]))
                continue
        # domination par une série convergente indépendante de x (M_n = sup |u_n|)
        if isinstance(ens, sp.Interval):
            cd = _cv_candidats_serie(u_n, n, x, ens)
            if cd is not None and not any(c == sp.oo for c in cd[0]):
                ok_dom = True
                for c in [c for c in cd[0] if c != 0]:
                    v, lg = _cv_serie_numerique(c, n, n_min)
                    if v is not True:
                        ok_dom = False
                        break
                if ok_dom:
                    maj = [c for c in cd[0] if c != 0]
                    sortie.append(dict(ens=ens, nature="cv_abs", geo=None, lignes=[
                        pre + r"|u_n(x)|\le M_n=" + (sp.latex(maj[0]) if len(maj) == 1 else r"\max\left(" + r",\ ".join(sp.latex(c) for c in maj) + r"\right)")
                        + (_cv_motif_serie(maj[0], n) if len(maj) == 1 else r"\text{ ; chaque série } \sum M_n \text{ converge}") + r"\text{ ; par comparaison } \sum u_n(x) \text{ converge absolument.}"]))
                    continue
        # séries alternées
        lb = _cv_leibniz_serie(u_n, n, x, ens) if isinstance(ens, sp.Interval) else None
        if lb is not None:
            a, d = lb
            sortie.append(dict(ens=ens, nature="cv", geo=None, lignes=[
                pre + r"u_n(x)=(-1)^n a_n(x),\ a_n(x)=" + sp.latex(a)
                + r"\ge 0,\quad a_{n+1}(x)-a_n(x)=" + sp.latex(d) + r"\le 0,\quad a_n(x)\to 0\text{ : critère de Leibniz, } \sum u_n(x)\text{ converge.}"]))
            continue
        sortie.append(dict(ens=ens, nature="ind", geo=None, lignes=[
            pre + r"\text{le calcul symbolique ne conclut pas sur la convergence simple de } \sum u_n(x).\text{}"]))
    return sortie




def _cv_candidats_serie(u_n, n, x, K):
    """Candidats pour M_n = sup_K |u_n|. Renvoie (candidats, exact, lignes) ou None."""
    lignes = []
    trig = u_n.has(sp.sin) or u_n.has(sp.cos) or u_n.has(sp.tan)
    latK = _cv_ens_latex(K)
    if trig:
        r_h, borne = _cv_sep_borne(u_n, n)
        if not borne or r_h.has(sp.sin) or r_h.has(sp.cos) or r_h.has(sp.tan):
            return None
        lignes.append(
            r"\text{Pour tout } x\in " + latK + r",\quad |u_n(x)|=\left|" + sp.latex(u_n)
            + r"\right|\le \left|" + sp.latex(r_h) + r"\right|\quad \text{car les facteurs trigonométriques de } " + sp.latex(sp.simplify(u_n / r_h)) + r" \text{ sont bornés par 1 en valeur absolue.}"
        )
        if r_h.has(x):
            r = _cv_sup_candidats(r_h, n, x, K)
            if r is None:
                return None
            return r[0], False, lignes
        return [sp.simplify(sp.Abs(r_h))], False, lignes
    r = _cv_sup_candidats(u_n, n, x, K)
    if r is None:
        return None
    cands, crits = r
    lignes.append(r"\text{Sur } " + latK + r",\quad u_n'(x)=" + sp.latex(sp.simplify(sp.diff(u_n, x))))
    if crits:
        lignes.append(r"\text{Cette dérivée s'annule en } x=" + r",\ ".join(sp.latex(c) for c in crits)
                      + r"\text{ (point critique situé dans } " + latK + r"\text{ pour } n \text{ assez grand).}")
    else:
        lignes.append(r"\text{Cette dérivée ne s'annule pas à l'intérieur de } " + latK + r"\text{ : } u_n \text{ y est monotone, son maximum en valeur absolue est atteint aux bornes.}")
    lv = _cv_ligne_valeurs("u_n(x)", n)
    if lv:
        lignes.append(lv)
    return cands, True, lignes


def _cv_normale(u_n, n, x, K, n_min):
    """Convergence normale sur K. Renvoie dict(verdict, lignes, cands, exact)."""
    latK = _cv_ens_latex(K)
    res = dict(verdict=None, lignes=[], cands=None, exact=False)
    c = _cv_candidats_serie(u_n, n, x, K)
    if c is None:
        res["lignes"].append(r"\text{Le calcul symbolique ne permet pas de déterminer } M_n=\sup_{x\in " + latK + r"}|u_n(x)|\text{.}")
        return res
    cands, exact, lignes = c
    res["cands"], res["exact"] = cands, exact
    res["lignes"] += lignes
    utiles = [v for v in cands if v != 0] or [sp.Integer(0)]
    signe = r"=" if exact else r"\le "
    if len(utiles) == 1:
        res["lignes"].append(r"M_n=\sup_{x\in " + latK + r"}|u_n(x)|" + signe + sp.latex(utiles[0]))
    else:
        res["lignes"].append(r"M_n=\sup_{x\in " + latK + r"}|u_n(x)|" + signe + r"\max\left(" + r",\ ".join(sp.latex(v) for v in utiles) + r"\right)")
    if any(v == sp.oo for v in utiles):
        res["lignes"].append(r"\text{La fonction } u_n \text{ n'est pas bornée sur } " + latK + r"\text{ : il n'y a pas convergence normale sur } " + latK + r".")
        res["verdict"] = False
        return res
    utiles = _cv_retire_equivalents(utiles, n)
    verdicts = []
    for v in utiles:
        if v == 0:
            verdicts.append(True)
            continue
        vv, lg = _cv_serie_numerique(v, n, n_min, memo=True)
        if isinstance(lg, _CvRep):
            res["lignes"].append(r"\text{Pour } M_n \text{ : }" + lg[0])
        else:
            res["lignes"].append(r"\text{Étude de la série numérique } \sum " + sp.latex(v) + r"\text{ :}")
            res["lignes"] += lg
        verdicts.append(vv)
    if all(v is True for v in verdicts):
        res["lignes"].append(r"\text{La série } \sum M_n \text{ converge : } \sum u_n \text{ converge normalement sur } " + latK + r".")
        res["verdict"] = True
    elif exact and any(v is False for v in verdicts):
        res["lignes"].append(
            r"\text{Comme } M_n \ge " + sp.latex(next(u for u, v in zip(utiles, verdicts) if v is False))
            + r" \text{ et que la série de ce minorant diverge, } \sum M_n \text{ diverge : pas de convergence normale sur } " + latK + r"."
        )
        res["verdict"] = False
    else:
        res["lignes"].append(r"\text{Le calcul symbolique ne permet pas de conclure sur la convergence de } \sum M_n\text{.}")
    return res


def _cv_retire_equivalents(cands, n):
    """Retire les candidats dont le rapport à un autre candidat tend vers une constante >0 (même nature de série)."""
    gardes = []
    try:
        npos = sp.Symbol("n", integer=True, positive=True)
        for v in cands:
            doublon = False
            for k, w in enumerate(gardes):
                r = sp.limit(sp.simplify((v / w).subs(n, npos)), npos, sp.oo)
                if r.is_number and r.is_finite and r > 0:
                    doublon = True
                    if float(v.subs(n, 1)) > float(w.subs(n, 1)):
                        gardes[k] = v
                    break
            if not doublon:
                gardes.append(v)
        return gardes or cands
    except Exception:
        return cands


def _cv_uniforme_serie(u_n, n, x, K, n_min, normale):
    """Convergence uniforme de la série sur K. Renvoie dict(verdict, lignes)."""
    latK = _cv_ens_latex(K)
    res = dict(verdict=None, lignes=[])
    if normale["verdict"] is True:
        res["lignes"].append(r"\text{La convergence normale entraîne la convergence uniforme (théorème de Weierstrass) :}")
        res["lignes"].append(r"\sum u_n \text{ converge uniformément sur } " + latK + r".")
        res["verdict"] = True
        return res
    # condition nécessaire : (u_n) converge uniformément vers 0
    if normale["exact"] and normale["cands"] is not None:
        v, mauvais = _cv_verdict_candidats([c for c in normale["cands"] if c != 0] or [sp.Integer(0)], n)
        if v == "non":
            res["lignes"].append(
                r"\text{Si } \sum u_n \text{ convergeait uniformément sur } " + latK
                + r"\text{, alors } u_n=R_{n-1}-R_n \text{ convergerait uniformément vers } 0\text{.}"
            )
            if mauvais == sp.oo:
                res["lignes"].append(
                    r"\text{Or } u_n \text{ n'est pas bornée sur } " + latK
                    + r"\text{ : } (u_n) \text{ ne converge pas uniformément vers } 0 \text{ et } \sum u_n \text{ ne converge pas uniformément sur } " + latK + r"."
                )
            else:
                res["lignes"].append(
                    r"\text{Or } \sup_{x\in " + latK + r"}|u_n(x)|\ge " + sp.latex(mauvais)
                    + r"\text{ ne tend pas vers } 0\text{ : } \sum u_n \text{ ne converge pas uniformément sur } " + latK + r"."
                )
            res["verdict"] = False
            return res
    t = _cv_suites_test(lambda p: u_n, [dict(ens=K, statut="cv", lim=sp.Integer(0))], n, x, K)
    if t is not None:
        xn, ecart, L = t
        res["lignes"].append(
            r"\text{Posons } x_n=" + sp.latex(xn) + r"\in " + latK + r"\text{. Alors } |u_n(x_n)|=" + sp.latex(ecart)
            + r"\longrightarrow " + sp.latex(L) + r"\ne 0\text{.}"
        )
        res["lignes"].append(
            r"\text{Donc } (u_n) \text{ ne converge pas uniformément vers } 0 \text{ sur } " + latK
            + r"\text{ : } \sum u_n \text{ ne converge pas uniformément sur } " + latK + r"."
        )
        res["verdict"] = False
        return res
    # séries alternées : |R_n(x)| <= a_{n+1}(x)
    if isinstance(K, sp.Interval):
        lb = _cv_leibniz_serie(u_n, n, x, K)
        if lb is not None:
            a, d = lb
            if a.has(x):
                r = _cv_sup_candidats(a, n, x, K)
                cands = r[0] if r is not None else None
            else:
                cands = [a]
            if cands is not None:
                v, _ = _cv_verdict_candidats([c for c in cands if c != 0] or [sp.Integer(0)], n)
                if v == "zero":
                    res["lignes"].append(
                        r"\text{On a } u_n(x)=(-1)^n a_n(x) \text{ avec } a_n(x)=" + sp.latex(a)
                        + r"\ge 0 \text{ décroissante en } n \text{, donc pour tout } x\in " + latK + r",\ |R_n(x)|\le a_{n+1}(x)\le \sup_{" + latK + r"}a_{n+1}."
                    )
                    res["lignes"].append(
                        r"\text{Or } \sup a_{n} \longrightarrow 0\text{ : } \sum u_n \text{ converge uniformément sur } " + latK + r"\text{ (critère de Leibniz uniforme).}"
                    )
                    res["verdict"] = True
                    return res
    res["lignes"].append(r"\text{Le calcul symbolique ne permet pas de conclure sur la convergence uniforme sur } " + latK + r"\text{.}")
    return res


def _cv_lignes_segments_serie(pieces_serie, n, x):
    """Convergence normale sur tout segment d'un intervalle où |b|<1. Renvoie (lignes, [J])."""
    fake = [dict(geo=p["geo"], statut="cv", ens=p["ens"]) for p in pieces_serie
            if p.get("geo") is not None and p["nature"] == "cv_abs"]
    lignes, zones = [], []
    for J, b, q in _cv_segments_geo(fake, n, x):
        lat = _cv_ens_latex(J)
        zones.append(J)
        lb = sp.latex(b)
        q = sp.simplify(q)
        try:
            dom = _cv_terme_dominant(q, n) if q.is_rational_function(n) else None
        except Exception:
            dom = None
        lignes.append(
            r"\text{Soit } [\alpha,\beta]\subset " + lat + r"\text{ un segment quelconque. } x\mapsto \left|" + lb
            + r"\right| \text{ est continue sur ce segment fermé donc atteint son maximum } \rho=\max_{[\alpha,\beta]}\left|" + lb
            + r"\right|<1\text{ (car ce maximum est atteint en un point de } " + lat + r"\text{).}"
        )
        if sp.simplify(q - 1) == 0:
            mn = r"M_n \le \rho^{n}"
            suite = r"\sum \rho^n \text{ est une série géométrique de raison } \rho<1\text{, donc convergente.}"
        elif not q.has(x):
            rq = sp.simplify(q.subs(n, n + 1) / q)
            mn = r"|u_n(x)|=|" + sp.latex(q) + r"|\cdot\left|" + lb + r"\right|^n\le " + sp.latex(q) + r"\,\rho^{n}"
            suite = (r"\text{pour la série } \sum " + sp.latex(q) + r"\rho^n \text{ : } \frac{(n+1)\rho^{n+1}}{n\rho^{n}}\text{ s'écrit } " + sp.latex(rq) + r"\,\rho\longrightarrow\rho<1"
                     + r"\text{, donc par d'Alembert elle converge.}")
        else:
            kk = sp.latex(dom[0]) if dom else r"k"
            mn = r"|u_n(x)|\le |" + sp.latex(q) + r"|\,\rho^{n}\text{, avec } |" + sp.latex(q) + r"|\le C\,n^{" + kk + r"}\ (C \text{ constante indépendante de } n \text{ et } x)"
            suite = (r"\text{le rapport de deux termes consécutifs de } n^{" + kk + r"}\rho^n \text{ vaut } \left(1+\frac1n\right)^{" + kk
                     + r"}\rho\to\rho<1\text{ : par d'Alembert } \sum n^{" + kk + r"}\rho^n \text{ converge.}")
        lignes.append(
            r"\text{Pour tout } x\in[\alpha,\beta]:\quad " + mn + r"\text{ ; } " + suite
        )
        lignes.append(
            r"\text{Donc } \sum u_n \text{ converge normalement, donc uniformément, sur tout segment inclus dans } " + lat + r"."
        )
    return lignes, zones


def _cv_segments_symboliques(u_n, n, x, K, n_min):
    """Normale sur tout segment : segment [-a, a] ou [0, a] avec a > 0 symbolique (K = R ou [0, oo))."""
    a = sp.Symbol("a", positive=True)
    try:
        if K == S.Reals:
            seg = sp.Interval(-a, a)
        elif isinstance(K, sp.Interval) and K.start == 0 and K.end == sp.oo:
            seg = sp.Interval(0, a)
        else:
            return [], []
        c = _cv_candidats_serie(u_n, n, x, seg)
        if c is None:
            return [], []
        cands = [v for v in c[0] if v != 0]
        if not cands or any(v == sp.oo for v in cands):
            return [], []
        for v in cands:
            vv, _ = _cv_serie_numerique(v, n, n_min)
            if vv is not True:
                return [], []
        latK = _cv_ens_latex(K)
        maj = cands[0] if len(cands) == 1 else None
        l1 = (r"\text{Sur tout segment } " + sp.latex(seg) + r"\ (a>0)\text{ : } M_n(a)=\sup_{|x|\le a}|u_n(x)|"
              + (r"=" + sp.latex(maj) if maj is not None else r"\le \max\left(" + r",\ ".join(sp.latex(v) for v in cands) + r"\right)")
              + (_cv_motif_serie(maj, n).replace("M_{n+1}", "@@1").replace("M_n", "M_n(a)").replace("@@1", "M_{n+1}(a)") if maj is not None else r"\text{ ; chaque série } \sum M_n(a) \text{ converge}") + r"\text{.}")
        l2 = r"\text{Donc } \sum u_n \text{ converge normalement, donc uniformément, sur tout segment inclus dans } " + latK + r"."
        return [l1, l2], [K]
    except Exception:
        return [], []


def _cv_serie_analyse(u_n, n, x, I, n_min, profondeur=1):
    """Étude complète d'une série de fonctions. Renvoie un dictionnaire de lignes et de verdicts."""
    res = dict(simple=[], normale=[], unif=[], cons=[], v_simple=None, v_normale=None, v_unif=None, zones=[], E=None)
    if profondeur > 0:
        _CV_MEMO.clear()
    latI = _latex_intervalle(I)
    pieces = _cv_serie_pieces(u_n, n, x, I, n_min)
    for p in pieces:
        res["simple"] += p["lignes"]
    cv = [p for p in pieces if p["nature"] in ("cv", "cv_abs")]
    non_cv = [p for p in pieces if p["nature"] not in ("cv", "cv_abs")]
    if cv:
        res["E"] = sp.Union(*[p["ens"] for p in cv])
    if not non_cv:
        res["v_simple"] = True
        res["simple"].append(r"\text{Ainsi } \sum u_n \text{ converge simplement sur } " + latI + r".")
    else:
        res["v_simple"] = False if all(p["nature"] == "div" for p in non_cv) else None
        if cv:
            res["simple"].append(r"\text{Ainsi } \sum u_n \text{ converge simplement sur } " + sp.latex(res["E"]) + r".")
        div = [p for p in non_cv if p["nature"] == "div"]
        if div:
            res["simple"].append(
                r"\text{La série diverge pour } x\in " + sp.latex(sp.Union(*[p["ens"] for p in div]))
                + r"\text{ : elle ne converge pas simplement sur } " + latI + r"."
            )
        ind = [p for p in non_cv if p["nature"] == "ind"]
        if ind:
            res["simple"].append(
                r"\text{Le calcul symbolique ne permet pas de conclure pour } x\in "
                + sp.latex(sp.Union(*[p["ens"] for p in ind])) + r"\text{.}"
            )
    absolue = [p for p in pieces if p["nature"] == "cv_abs"]
    if absolue and not non_cv and len(absolue) == len(pieces):
        res["simple"].append(r"\text{La convergence est même absolue en tout point de } " + latI + r".")

    # domaine d'étude de l'uniformité : l'ensemble de convergence simple, s'il est un intervalle
    K = I if res["v_simple"] is True else res["E"]
    if K is not None and K != I:
        if isinstance(K, sp.Interval) or K == S.Reals:
            res["simple"].append(r"\text{On étudie la convergence normale et uniforme sur l'ensemble de convergence simple } E=" + sp.latex(K) + r".")
        else:
            K = None
    if K is None or isinstance(K, sp.Union):
        res["normale"].append(r"\text{L'ensemble de convergence simple n'est pas un intervalle : l'étude de la convergence normale n'est pas menée.}")
        seg_l, zones = _cv_lignes_segments_serie(pieces, n, x)
        res["normale"] += seg_l
        res["zones"] = zones
        return res

    # ---------------- convergence normale
    nor = _cv_normale(u_n, n, x, K, n_min)
    res["normale"] += nor["lignes"]
    res["v_normale"] = nor["verdict"]
    # ---------------- convergence uniforme
    un = _cv_uniforme_serie(u_n, n, x, K, n_min, nor)
    res["unif"] += un["lignes"]
    res["v_unif"] = un["verdict"]
    # ---------------- sur tout segment
    if res["v_normale"] is not True:
        seg_l, zones = _cv_lignes_segments_serie(pieces, n, x)
        if not zones:
            seg_l, zones = _cv_segments_symboliques(u_n, n, x, K, n_min)
        res["zones"] = zones
        if res["v_unif"] is not True:
            res["unif"] += seg_l
    latK = _cv_ens_latex(K)

    # ---------------- propriétés de la somme
    cont = _cv_continue_sur(u_n, x, K, n)
    if res["v_unif"] is True and cont:
        res["cons"].append(
            r"\text{Chaque } u_n \text{ est continue sur } " + latK
            + r" \text{ et } \sum u_n \text{ converge uniformément : la somme } S=\sum_{n\ge " + sp.latex(n_min) + r"} u_n \text{ est continue sur } " + latK + r"."
        )
        if _cv_est_segment(K):
            res["cons"].append(
                r"\text{Intégration terme à terme sur le segment : } \int_{" + sp.latex(K.start) + r"}^{" + sp.latex(K.end)
                + r"} S(x)\,dx=\sum_{n\ge " + sp.latex(n_min) + r"}\int_{" + sp.latex(K.start) + r"}^{" + sp.latex(K.end) + r"} u_n(x)\,dx."
            )
    elif res["zones"] and cont:
        for J in res["zones"]:
            res["cons"].append(
                r"\text{Chaque } u_n \text{ est continue et } \sum u_n \text{ converge uniformément sur tout segment de } " + _cv_ens_latex(J)
                + r"\text{ : la somme } S \text{ est continue sur } " + _cv_ens_latex(J) + r"."
            )
    if profondeur > 0 and (res["v_unif"] is True or res["zones"]) and cont:
        try:
            up = sp.simplify(sp.diff(u_n, x))
            if _cv_continue_sur(up, x, K, n):
                sous = _cv_serie_analyse(up, n, x, K, n_min, profondeur=0)
                if res["v_unif"] is True and sous["v_unif"] is True:
                    res["cons"].append(
                        r"\text{Dérivation terme à terme : } u_n'(x)=" + sp.latex(up)
                        + r"\text{ et } \sum u_n' \text{ converge uniformément sur } " + latK
                        + r"\text{ : } S \text{ est de classe } C^1 \text{ et } S'=\sum u_n'."
                    )
                elif sous["zones"]:
                    for J in sous["zones"]:
                        res["cons"].append(
                            r"\text{Dérivation terme à terme : } \sum u_n' \text{ converge uniformément sur tout segment de } " + _cv_ens_latex(J)
                            + r"\text{ : } S \text{ est de classe } C^1 \text{ sur } " + _cv_ens_latex(J) + r" \text{ et } S'=\sum u_n'."
                        )
        except Exception:
            pass
    return res


_CV_VUS = set()


def _cv_affiche(titre, lignes):
    """Affiche une section de l'étude de convergence (titre puis lignes LaTeX)."""
    if not lignes:
        return
    st.latex(_cv_titre(titre))
    for l in lignes:
        if l in _CV_VUS:
            continue
        _CV_VUS.add(l)
        st.latex(l)


def etude_suite_de_fonctions(f_n, n, x, intervalle, n_min=1, n_max=20):
    if _intervalle_est_vide(intervalle):
        st.latex(_latex_rouge("L’intervalle est vide."))
        return

    try:
        f_n = sp.simplify(sp.sympify(f_n))
    except Exception:
        st.latex(_latex_rouge("Expression invalide pour f_n."))
        return

    _CV_VUS.clear()
    latex_I = _latex_intervalle(intervalle)

    st.latex(r"\text{Suite de fonctions définie par }\quad f_n(x)=" + sp.latex(f_n) + r"\quad \text{sur }" + latex_I)

    f_lim, ok = _limite_simple(f_n, n)
    if not ok:
        f_lim = None

    # --- Convergences simple et uniforme (programme de L2) ---
    detail = False
    try:
        etude = _cv_avec_delai(lambda: _cv_suite_analyse(f_n, n, x, intervalle, int(n_min)), 60)
        if etude is not None and etude["simple"]:
            detail = True
            _cv_affiche("Convergence simple", etude["simple"])
            _cv_affiche("Convergence uniforme", etude["unif"])
            _cv_affiche("Conséquences de la convergence uniforme", etude["cons"])
    except BaseException as e:
        if isinstance(e, (SystemExit, GeneratorExit)):
            raise
    if not detail:
        if ok:
            st.latex(r"\forall x\in " + latex_I + r",\quad \lim_{n\to+\infty} f_n(x)=" + sp.latex(f_lim))
        else:
            st.latex(r"\text{La limite simple de la suite }(f_n)\text{ n’existe pas.}")

    _plot_famille_et_limite(f_n, f_lim, x, intervalle, n, n_min=int(n_min), n_max=int(n_max))


def etude_serie_de_fonctions(u_n, n, x, intervalle, n_min=1, N_graphe=20):
    if _intervalle_est_vide(intervalle):
        st.latex(_latex_rouge("L’intervalle est vide."))
        return

    try:
        u_n = sp.simplify(sp.sympify(u_n))
    except Exception:
        st.latex(_latex_rouge("Expression invalide pour u_n(x)."))
        return

    _CV_VUS.clear()
    latex_I = _latex_intervalle(intervalle)
    n_min_int = int(n_min)
    N_graphe_int = int(N_graphe)

    st.latex(
        r"\text{Série de fonctions }\sum_{n\ge " + sp.latex(n_min_int) + r"} u_n(x),\quad "
        r"u_n(x)=" + sp.latex(u_n) + r"\quad \text{sur }" + latex_I
    )

    # --- Convergences simple, normale et uniforme (programme de L2) ---
    detail = False
    try:
        etude = _cv_avec_delai(lambda: _cv_serie_analyse(u_n, n, x, intervalle, n_min_int), 90)
        if etude is not None and etude["simple"]:
            detail = True
            _cv_affiche("Convergence simple de la série", etude["simple"])
            _cv_affiche("Convergence normale", etude["normale"])
            _cv_affiche("Convergence uniforme", etude["unif"])
            _cv_affiche("Conséquences sur la somme", etude["cons"])
    except BaseException as e:
        if isinstance(e, (SystemExit, GeneratorExit)):
            raise
    if not detail:
        u_lim, ok = _limite_simple(u_n, n)
        if ok:
            st.latex(r"\forall x\in " + latex_I + r",\quad \lim_{n\to+\infty} u_n(x)=" + sp.latex(u_lim))
        else:
            st.latex(r"\text{La limite simple du terme général }u_n(x)\text{ n’existe pas.}")
        st.latex(r"\text{L’étude détaillée des convergences simple, normale et uniforme n’a pas pu être menée automatiquement.}")

    _plot_series_partielles(u_n, x, intervalle, n, n_min=n_min_int, N_max=max(n_min_int, N_graphe_int))


# =============================================================
# 6) Série entière : graphes + reconnaissance a_n(z(x))^n
# =============================================================
def _trace_sommes_partielles_et_limite(u_n_x, x, intervalle, n, n_min=0, N_max=20, f_lim=None):
    xmin, xmax = _bornes_intervalle_pour_graphe(intervalle)
    xs = np.linspace(xmin, xmax, 1600)

    fig, ax = plt.subplots(figsize=(18, 10))
    cmap = plt.get_cmap("tab20")

    Y_MAX = 20 # limite verticale d'affichage

    n_min = int(n_min)
    N_max = int(N_max)
    if N_max < n_min:
        N_max = n_min

    for N in range(n_min, N_max + 1):
        Sx = np.zeros_like(xs, dtype=float)
        for j in range(n_min, N + 1):
            try:
                terme_j = sp.simplify(u_n_x.subs(n, j))
                Sx += _eval_vecteur(x, terme_j, xs)
            except Exception:
                pass

        # Écrêtage VISUEL uniquement
        Sx = np.clip(Sx, -Y_MAX, Y_MAX)

        ax.plot(
            xs, Sx,
            color=cmap((N - n_min) % 20),
            linewidth=1.7,
            alpha=0.98,
            label=rf"$S_{{{N}}}(x)$"
        )

    if f_lim is not None:
        y_lim = _eval_vecteur(x, f_lim, xs)

        # Écrêtage VISUEL de la limite aussi
        y_lim = np.clip(y_lim, -Y_MAX, Y_MAX)

        ax.plot(
            xs, y_lim,
            color="black",
            linewidth=3.2,
            label=r"$f(x)$"
        )

    ax.set_xlabel("x")
    ax.set_ylabel("Sommes partielles")
    ax.set_ylim(-Y_MAX, Y_MAX)
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.25)

    st.pyplot(fig)


def _extraire_forme_puissance(expr, n, x):
    expr = sp.simplify(expr)

    try:
        for p in expr.atoms(sp.Pow):
            base, exp_ = p.as_base_exp()
            if sp.simplify(exp_ - n).equals(0) and base.has(x) and (not base.has(n)):
                a_n = sp.simplify(expr / p)
                return sp.simplify(a_n), sp.simplify(base)
    except Exception:
        pass

    try:
        if expr.has(x**n):
            a_n = sp.simplify(expr / (x**n))
            return sp.simplify(a_n), x
    except Exception:
        pass

    return None, None


# =============================================================
# 7) Rayon : Cauchy–Hadamard (robuste sans limit_sup)
# =============================================================
def _limsup_racine_absolue(a_n, n):
    """
    Renvoie limsup (|a_n|)^(1/n) si SymPy arrive à donner une borne d'accumulation.
    Utilise limit_seq, qui peut renvoyer AccumulationBounds(min,max).
    """
    try:
        from sympy.calculus.accumulationbounds import AccumulationBounds
        seq = sp.Abs(a_n) ** (sp.Rational(1, 1) / n)
        L = sp.limit_seq(seq, n)
        if L is None or isinstance(L, sp.Limit):
            return None
        if isinstance(L, AccumulationBounds):
            return sp.simplify(L.max)
        return sp.simplify(L)
    except Exception:
        return None


def _lim_racine_absolue(a_n, n):
    try:
        seq = sp.Abs(a_n) ** (sp.Rational(1, 1) / n)
        L = sp.limit_seq(seq, n)
        if L is None or isinstance(L, sp.Limit):
            return None
        return sp.simplify(L)
    except Exception:
        return None


def _rayon_depuis_L(L):
    if L is None:
        return None
    try:
        if sp.simplify(L).equals(0):
            return sp.oo
        if L == sp.oo:
            return sp.Integer(0)
        return sp.simplify(1 / L)
    except Exception:
        return None


import sympy as sp
import sympy as sp

import sympy as sp

import sympy as sp

import sympy as sp

def _extraire_forme_puissance(expr, n, x):
    try:
        expr = sp.powsimp(sp.expand_power_exp(expr), combine='all')
        
        if not expr.has(x):
            return expr, sp.Integer(1)

        res_an, res_z = None, None

        a_n_simple = sp.simplify(expr / (x**n))
        if not a_n_simple.has(x):
            res_an, res_z = a_n_simple, x

        if res_an is None:
            for p in expr.atoms(sp.Pow):
                base, exp_ = p.as_base_exp()
                if not base.has(x) or base.has(n):
                    continue

                exp_s = sp.simplify(exp_)
                k_mul = sp.simplify(exp_s / n)

                if k_mul.is_integer and not k_mul.equals(0):
                    try:
                        c = facteur2(base)
                    except:
                        c = base.as_coeff_Mul()[0]

                    z_raw = sp.simplify(base / c)
                    
                    k_z = 1
                    if z_raw.is_Pow:
                        b, e = z_raw.as_base_exp()
                        if b == x and e.is_integer and e > 0:
                            k_z = e
                        else: continue
                    elif z_raw == x:
                        k_z = 1
                    else:
                        continue

                    a_n = sp.simplify((expr / p) * (c**exp_s))
                    if not a_n.has(x):
                        res_an, res_z = a_n, x**(k_z * k_mul)
                        break

        if res_an is None:
            try:
                coeff_indep, x_part = expr.as_independent(x, as_Add=False)
                if x_part.has(x):
                    expo_x = sp.simplify(sp.log(x_part, x))
                    k_final = sp.simplify(expo_x / n)
                    if not k_final.has(x) and k_final > 0:
                        res_an, res_z = coeff_indep, x**k_final
            except:
                pass

        if res_an is not None and res_z is not None:
            try:
                base_z, exp_z = res_z.as_base_exp()
                try:
                    c_final = facteur2(base_z)
                except:
                    c_final = base_z.as_coeff_Mul()[0]
                
                if c_final != 1:
                    res_an = sp.simplify(res_an * (c_final**(exp_z * n)))
                    res_z = sp.simplify((base_z / c_final)**exp_z)
            except:
                pass
            return sp.simplify(res_an), res_z

    except:
        pass
    
    return None, None



def _rayon_cauchy_hadamard(a_n, n):
    Lsimple = _lim_racine_absolue(a_n, n)
    Lsup = _limsup_racine_absolue(a_n, n)
    R = _rayon_depuis_L(Lsup if Lsup is not None else Lsimple)
    return R, Lsimple, Lsup


def _rayon_dalembert(a_n, n):
    try:
        q = sp.simplify(sp.Abs(a_n.subs(n, n + 1) / a_n))
        L = sp.limit(q, n, sp.oo)
        if L is None or isinstance(L, sp.Limit):
            return None, None
        L = sp.simplify(L)
        if sp.simplify(L).equals(0):
            return sp.oo, L
        if L == sp.oo:
            return sp.Integer(0), L
        return sp.simplify(1 / L), L
    except Exception:
        return None, None


# =============================================================
# 8) Somme fermée (si possible)
# =============================================================
def _nettoyer_expression_somme(expr):
    try:
        e = sp.simplify(expr)
    except Exception:
        e = expr

    try:
        if isinstance(e, sp.Piecewise) and len(e.args) >= 1:
            e = e.args[0][0]
    except Exception:
        pass

    try:
        e = e.rewrite(sp.factorial)
    except Exception:
        pass

    try:
        e = sp.hyperexpand(e)
    except Exception:
        pass

    try:
        e = sp.simplify(e)
    except Exception:
        pass

    try:
        if e.has(sp.I):
            e = sp.re(e)
            e = sp.simplify(e)
    except Exception:
        pass

    return e


def _serie_vers_fonction(a_n, n, z, n_min=0):
    try:
        t = sp.Symbol("t", real=True)
        S_formelle = sp.Sum(a_n * (t ** n), (n, int(n_min), sp.oo))
        f_t = S_formelle.doit()
        f_t = _nettoyer_expression_somme(f_t)
        f_z = sp.simplify(f_t.subs(t, z))
        return f_z
    except Exception:
        return None


def _ratio_geometrique(u_n_x, n, x):
    try:
        q = sp.simplify(u_n_x.subs(n, n + 1) / u_n_x)
        if (not q.has(n)) and q.has(x):
            return sp.simplify(q)
        return None
    except Exception:
        return None


# =============================================================
# 9) Étude : série entière / série de puissances sur I
# =============================================================
def etude_serie_entiere_convergence(expr_terme, n, x, intervalle, n_min=1, N_graphe=20):
    # ==========================================================
    # 0) Vérification de l’intervalle
    # ==========================================================
    if _intervalle_est_vide(intervalle):
        st.latex(_latex_rouge(
            "L’intervalle d’étude choisi est vide. Veuillez corriger l’intervalle."
        ))
        return

    # ==========================================================
    # 1) Lecture et validation du terme général
    # ==========================================================
    try:
        u_n_x = sp.simplify(sp.sympify(expr_terme))
    except Exception:
        st.latex(_latex_rouge(
            "L’expression du terme général est invalide. Veuillez la corriger."
        ))
        return

    try:
        n_min = int(n_min)
        N_graphe = int(N_graphe)
    except Exception:
        st.latex(_latex_rouge(
            "L’indice de départ doit être un entier valide."
        ))
        return

    latex_I = _latex_intervalle(intervalle)

    # ==========================================================
    # 2) Domaine de définition en n (INDICE)
    # ==========================================================
    try:
        Dom_n = sp.calculus.util.continuous_domain(u_n_x, n, S.Reals)
        if not Dom_n.contains(n_min):
            st.latex(_latex_rouge(
                "La série n’est pas définie pour l’indice de départ choisi. "
                "Veuillez corriger l’indice de départ."
            ))
            return
    except Exception:
        pass
    # ==========================================================
    # 3) Domaine de définition en x (VARIABLE)
    # ==========================================================
    try:
        Dom_x = sp.calculus.util.continuous_domain(u_n_x, x, S.Reals)
        if intervalle != S.Reals:
            intervalle_eff = intervalle.intersect(Dom_x)
        else:
            intervalle_eff = Dom_x

        if intervalle_eff == S.EmptySet:
            st.latex(_latex_rouge(
                "Aucune valeur de l’intervalle choisi ne respecte le domaine "
                "de définition de la fonction. Veuillez corriger l’intervalle."
            ))
            return
    except Exception:
        intervalle_eff = intervalle  # fallback sûr

    # ==========================================================
    # 4) Énoncé mathématique clair
    # ==========================================================
    st.latex(
        r"\text{On considère la série entière}\quad "
        r"\sum u_n(x)\quad \text{définie sur}\quad I=" + _latex_intervalle(intervalle_eff) + r"."
    )
    st.latex(
        r"\text{Le terme général est donné par}\quad u_n(x)=" + sp.latex(u_n_x) + r"."
    )

    # ==========================================================
    # 5) Reconnaissance série géométrique (accélération)
    # ==========================================================
    f_geo = None
    try:
        qx = _ratio_geometrique(u_n_x, n, x)
        if qx is not None:
            u0 = sp.simplify(u_n_x.subs(n, n_min))
            f_geo = sp.simplify(u0 / (1 - qx))

            st.latex(
                r"\text{Le rapport }\frac{u_{n+1}(x)}{u_n(x)}=" + sp.latex(qx)
                + r"\text{ est indépendant de l’indice.}"
            )
            st.latex(
                r"\text{La série est donc géométrique, et elle converge pour }"
                r"|" + sp.latex(qx) + r"|<1."
            )
    except Exception:
        f_geo = None

    # ==========================================================
    # 6) Mise sous la forme a_n (z(x))^n
    # ==========================================================
    a_n, z = _extraire_forme_puissance(u_n_x, n, x)

    if a_n is None or z is None:
        st.latex(
            r"\text{La série ne peut pas être mise sous la forme d’une série entière standard.}"
        )
        st.latex(r"\text{On procède néanmoins à une visualisation des sommes partielles.}")
        _trace_sommes_partielles_et_limite(
            u_n_x, x, intervalle_eff, n,
            n_min=n_min, N_max=max(N_graphe, n_min),
            f_lim=f_geo
        )
        return

    st.latex(
        r"\text{On écrit le terme général sous la forme}\quad "
        r"u_n(x)=a_n\,\bigl(z(x)\bigr)^n."
    )
    st.latex(r"u_n(x)" + r" =" + sp.latex(a_n) + sp.latex(z**n))
    # ==========================================================
    # 7) Calcul du rayon – Cauchy–Hadamard puis d’Alembert
    # ==========================================================
    R_prime = None

    Lsup = _limsup_racine_absolue(a_n, n)
    if Lsup is not None:
        st.latex(r"\text{On determine R' = } R(\sum a_nx^n) \text{ par la formule de Cauchy - Hadamard }")
        st.latex(
            r"\ell=\limsup_{n\to+\infty}\sqrt[n]{|a_n|}="+ r"\limsup_{n\to+\infty}\sqrt[n]{" +sp.latex(sp.Abs(a_n))+r"}=" + sp.latex(Lsup)
        )
        R_prime = _rayon_depuis_L(Lsup)
        st.latex(
            r"R'=\frac{1}{\ell}=" + sp.latex(R_prime)
        )
    else:
        L = _lim_racine_absolue(a_n, n)
        if L is not None:
            st.latex(r"\text{On determine R' = } R(\sum a_nz_n) \text{ par la formule de Cauchy - Hadamard }")
            st.latex(
                r"\lim_{n\to+\infty}\sqrt[n]{|a_n|}=" + sp.latex(L)
            )
            R_prime = _rayon_depuis_L(L)
            st.latex(
                r"R'=\frac{1}{\ell}=" + sp.latex(R_prime)
            )
        else:
            R2, Ld = _rayon_dalembert(a_n, n)
            if R2 is None:
                st.latex(_latex_rouge(
                    "Impossible de déterminer le rayon de convergence avec les critères classiques."
                ))
                return
            R_prime = R2
            st.latex(r"\text{On determine R' = } R(\sum a_nz^n) \text{ par le critère d'Alembert}")
            st.latex(
                r"\lim_{n\to+\infty}\left|\frac{a_{n+1}}{a_n}\right|="
                + sp.latex(Ld)
            )
            st.latex(
                r"R'=\frac{1}{\ell}=" + sp.latex(R_prime)
            )

    # ==========================================================
    # 8) Ajustement du rayon si z(x)=x^k avec k>1
    # ==========================================================
    k_val = 1
    try:
        if z.is_Pow and z.base == x and z.exp.is_integer and z.exp > 1:
            k_val = int(z.exp)
    except Exception:
        k_val = 1

    if k_val != 1:
        R = sp.simplify(R_prime ** (sp.Rational(1, k_val)))
        st.latex(
            r"\text{Puisque la variable apparaît sous la forme d’une puissance.}" + r"( z(x)= "+ sp.latex(z)+ r")"
        )
        st.latex(
            r"R=\sqrt[" + sp.latex(k_val) + r"]{R'}=" + sp.latex(R)
        )
    else:
        R = R_prime
        st.latex(
            r"\text{Le rayon de convergence est}\quad R=" + sp.latex(R)
        )

    # ==========================================================
    # 9) Conclusion générale (réelle et complexe)
    # ==========================================================
    st.latex(r"\text{Conclusion générale :}")
    st.latex(
        sp.latex(sp.Abs(x))+ r"<" + sp.latex(R)+ r"\ \Longleftrightarrow\ \text{convergence absolue de la série de terme général } a_n" + sp.latex(x**(k_val*n))
    )
    #"|z(x)|"+ "=" +
    st.latex(
        sp.latex(sp.Abs(x))+ r">" + sp.latex(R)+ r"\ \Longleftrightarrow\ \text{divergence grossière de la série de terme général } a_n" + sp.latex(x**(k_val*n))
    )
    st.latex(r"\text{Cela vaut aussi  pour tout complexe } z \text{ à la place du réel }x")

    st.latex(r"\begin{aligned}"
    r"&\text{Dans le plan complexe, la série entière de coefficients } (a_n) \text{ converge normalement (donc uniformément) }\\"
    r"&\text{sur tout disque fermé } \{ z \in \mathbb{C} \mid |z| \le r \},\ \forall r < "+ sp.latex(R)+
    r"\end{aligned}"
)

    # ==========================================================
    # 10) Ensembles sur l’intervalle effectif
    # ==========================================================
    try:
        E_int = sp.solveset(sp.Abs(x) < R, x, domain=S.Reals).intersect(intervalle_eff)
        E_ext = sp.solveset(sp.Abs(x) > R, x, domain=S.Reals).intersect(intervalle_eff)
        E_bord = sp.solveset(sp.Eq(sp.Abs(x), R), x, domain=S.Reals).intersect(intervalle_eff)

        st.latex(r"\text{Sur l'intervalle de départ I = }" + sp.latex(intervalle) + r"\text{ On a :}")

        if E_int != S.EmptySet:
            st.latex(r"\text{Convergence absolue sur }\ " + sp.latex(E_int) + r".")
        if E_ext != S.EmptySet:
            st.latex(r"\text{Divergence sur }\ " + sp.latex(E_ext) + r".")
        if E_bord != S.EmptySet:
            st.latex(
                r"\text{Sur le bord, une étude spécifique est nécessaire sur }\ "
                + sp.latex(E_bord) + r"."
            )
    except Exception:
        pass

    # ==========================================================
    # 11) Étude du bord via critères numériques
    # ==========================================================
    if R not in (0, sp.oo):
        try:
            bord = sp.solveset(sp.Eq(sp.Abs(x), R), x, domain=S.Reals).intersect(intervalle_eff)
            if isinstance(bord, sp.FiniteSet):
                i = 1
                for x0 in bord:
                    u_bord = sp.simplify(u_n_x.subs(x, x0))
                    S_bord = sp.Sum(u_bord.subs(n, k), (k, n_min, n))
                    
                    st.latex(sp.latex(i) +
                        r"\text{) Au point }" + sp.latex(x0)
                        + r",\ \text{on étudie la série numérique}"+ sp.latex(S_bord)
                    )
                    i = i + 1
                    try:
                        r_dec = critere(u_bord, n, n_min)
                        if r_dec is True:
                            st.latex(r"\text{Conclusion : convergence.}")
                        elif r_dec is False:
                            st.latex(r"\text{Conclusion : divergence.}")
                    except Exception:
                        pass
        except Exception:
            pass

    # ==========================================================
    # 12) Fonction somme (si possible)
    # ==========================================================
    f_lim = f_geo
    if f_lim is None:
        try:
            f_tmp = _serie_vers_fonction(a_n, n, z, n_min=n_min)
            if f_tmp is not None:
                f_lim = f_tmp
        except Exception:
            pass

    if f_lim is not None:
        st.latex(
            r"\text{Lorsque la série converge, elle définit la fonction somme}\quad "
            r"f(x)=" + sp.latex(f_lim) + r"."
        )

    # ==========================================================
    # 13) Graphe (TOUJOURS)
    # ==========================================================
    st.latex(r"\text{Visualisation des sommes partielles :}")
    _trace_sommes_partielles_et_limite(
        u_n_x, x, intervalle_eff, n,
        n_min=n_min, N_max=max(N_graphe, n_min),
        f_lim=f_lim
    )
# =============================================================
# 10) Développement en série entière
# =============================================================
def developpement_en_serie_entiere(f_expr, x, ordre=10):
    try:
        f = sp.simplify(sp.sympify(f_expr))
    except Exception:
        st.latex(_latex_rouge("Impossible : l’expression de la fonction est invalide."))
        return None, None, None

    ordre = int(ordre)
    st.latex(r"\text{On considère la fonction}\quad f(x)=" + sp.latex(f) + r".")

    somme = None
    n = sp.Symbol("n", integer=True, nonnegative=True)

    try:
        fps = sp.fps(f, x, 0)
        somme = fps.infinite
    except Exception:
        somme = None

    if somme is None:
        try:
            coeffs = [sp.simplify(sp.diff(f, x, k).subs(x, 0) / sp.factorial(k)) for k in range(0, ordre + 1)]
            somme = sp.Sum(coeffs[n] * x**n, (n, 0, ordre))
        except Exception:
            somme = None

    if somme is not None:
        st.latex(r"\text{Développement en série entière au voisinage de }0:\quad " + sp.latex(somme))

    ser_poly = None
    try:
        ser_poly = sp.series(f, x, 0, ordre + 1).removeO()
    except Exception:
        ser_poly = None

    R = None
    try:
        if isinstance(somme, sp.Sum):
            term = sp.simplify(somme.function)
            a_n = sp.simplify(term / (x**n))
            if not a_n.has(x):
                try:
                    R = sp.convergence_radius(a_n, n, x)
                except Exception:
                    R = None
    except Exception:
        R = None

    if R is not None:
        if R == sp.oo:
            st.latex(r"\text{Rayon de convergence : }R=+\infty\ \text{(validité sur }\mathbb{R}\text{).}")
        elif R == 0:
            st.latex(r"\text{Rayon de convergence : }R=0.")
        else:
            st.latex(r"\text{Rayon de convergence : }R=" + sp.latex(R) + r"\quad \Rightarrow\quad x\in(-R,R).")

    return f, ser_poly, R


def _trace_fonction_et_tronques(f, x, intervalle, ordre_max=10):
    xmin, xmax = _bornes_intervalle_pour_graphe(intervalle)
    xs = np.linspace(xmin, xmax, 1600)

    fig, ax = plt.subplots(figsize=(18, 5))

    y_f = _eval_vecteur(x, f, xs)
    ax.plot(xs, y_f, color="black", linewidth=3.2, label=r"$f(x)$")

    ordre_max = int(ordre_max)
    cmap = plt.get_cmap("tab20")
    for m in range(1, ordre_max + 1):
        try:
            tronque = sp.series(f, x, 0, m + 1).removeO()
            y_t = _eval_vecteur(x, tronque, xs)
            ax.plot(xs, y_t, color=cmap((m - 1) % 20), linewidth=1.6, alpha=0.98, label=rf"$T_{{{m}}}(x)$")
        except Exception:
            pass

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)


# =============================================================
# 11) Flask app (compatible app.js + index.html)
# =============================================================
app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/compute", methods=["POST"])
def compute():
    data = request.get_json(force=True) or {}
    out = []
    st.set_out(out)

    type_input = (data.get("type_input", "") or "").strip()
    user_input1 = (data.get("user_input1") or "").strip()
    user_input2 = (data.get("user_input2") or "").strip()

    try:
        # IMPORTANT : n doit pouvoir valoir 0 si la série commence à 0
        n = sp.symbols("n", integer=True, nonnegative=True)

        locals_dict = {
            "n": n,
            "pi": sp.pi,
            "e": sp.E,
            "E": sp.E,
            "I": sp.I,
            "sqrt": sp.sqrt,
            "abs": sp.Abs,
            "Abs": sp.Abs,
            "sign": sp.sign,
            "exp": sp.exp,
            "log": sp.log,
            "ln": sp.log,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "cot": sp.cot,
            "sec": sp.sec,
            "csc": sp.csc,
            "arcsin": sp.asin,
            "arccos": sp.acos,
            "arctan": sp.atan,
            "arccot": sp.acot,
            "arcsec": sp.asec,
            "arccsc": sp.acsc,
            "sinh": sp.sinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
            "coth": sp.acoth if hasattr(sp, "acoth") else None,
            "sech": sp.sech if hasattr(sp, "sech") else None,
            "csch": sp.csch if hasattr(sp, "csch") else None,
            "arsinh": sp.asinh,
            "arcosh": sp.acosh,
            "artanh": sp.atanh,
            "arcoth": sp.acoth if hasattr(sp, "acoth") else None,
            "arsech": sp.asech if hasattr(sp, "asech") else None,
            "arcsch": sp.acsch if hasattr(sp, "acsch") else None,
            "erf": sp.erf,
            "erfc": sp.erfc,
            "gamma": sp.gamma,
            "lgamma": sp.loggamma,
            "digamma": sp.digamma,
            "polygamma": sp.polygamma,
            "besselj": sp.besselj,
            "bessely": sp.bessely,
            "besseli": sp.besseli,
            "besselk": sp.besselk,
            "airyai": sp.airyai,
            "airybi": sp.airybi,
            "Heaviside": sp.Heaviside,
            "DiracDelta": sp.DiracDelta,
            "re": sp.re,
            "im": sp.im,
            "conjugate": sp.conjugate,
        }

        # Nettoyage des entrées None ajoutées conditionnellement
        locals_dict = {k: v for k, v in locals_dict.items() if v is not None}

        if type_input in ("Suites de fonctions", "Séries de fonctions", "Série entière", "Développement en série entière"):
            x = sp.symbols("x", real=True)
            locals_dict["x"] = x
            intervalle = _parse_interval_payload(data)

            if intervalle == S.EmptySet:
                st.latex(_latex_rouge("Impossible : l’intervalle choisi est vide."))
                return jsonify({"messages": out})

            if not user_input1:
                st.latex(_latex_rouge("Veuillez entrer une expression."))
                return jsonify({"messages": out})

            expr = sp.simplify(sp.sympify(user_input1, locals=locals_dict))

            if type_input == "Série entière":
                # app.js utilise user_input2 pour l'indice minimal
                n_min_raw = (data.get("n_min") or "").strip() or user_input2 or "0"
                try:
                    n_min = int(n_min_raw)
                except Exception:
                    st.latex(_latex_rouge("L’indice de départ doit être un entier."))
                    return jsonify({"messages": out})

                etude_serie_entiere_convergence(expr, n, x, intervalle, n_min=n_min, N_graphe=20)
                return jsonify({"messages": out})

            if type_input == "Développement en série entière":
                ordre = 10
                try:
                    if user_input2:
                        ordre = int(user_input2)
                except Exception:
                    ordre = 10

                f, ser, R = developpement_en_serie_entiere(expr, x, ordre=ordre)
                if f is not None:
                    if R is not None:
                        if R == sp.oo:
                            st.latex(r"\text{Intervalle de convergence (réel) : }\mathbb{R}.")
                        else:
                            st.latex(r"\text{Intervalle de convergence (réel) : }(-" + sp.latex(R) + r"," + sp.latex(R) + r").")
                    st.latex(r"\text{Représentation : la fonction et ses polynômes de Taylor.}")
                    _trace_fonction_et_tronques(f, x, intervalle, ordre_max=min(10, max(1, ordre)))
                return jsonify({"messages": out})

            if type_input == "Suites de fonctions":
                etude_suite_de_fonctions(expr, n, x, intervalle)
                return jsonify({"messages": out})

            if type_input == "Séries de fonctions":
                n_min_raw = (data.get("n_min") or "").strip() or user_input2
                if not n_min_raw:
                    st.latex(_latex_rouge("Veuillez renseigner l’indice de départ de la série."))
                    return jsonify({"messages": out})
                try:
                    n_min = int(n_min_raw)
                except Exception:
                    st.latex(_latex_rouge("L’indice de départ doit être un entier."))
                    return jsonify({"messages": out})
                etude_serie_de_fonctions(expr, n, x, intervalle, n_min)
                return jsonify({"messages": out})

        if type_input == "Série de Fourier":
            # Étude complète d'une série de Fourier (module fourier.py)
            try:
                import fourier as _fourier
            except Exception as _e:
                st.latex(_latex_rouge("Le module fourier.py est introuvable : placez-le à côté de app.py."))
                return jsonify({"messages": out})
            _fourier.st = st
            x = sp.symbols("x", real=True)
            locals_dict["x"] = x
            if not user_input1:
                st.latex(_latex_rouge("Veuillez entrer la fonction f(x) (périodique, donnée sur une période)."))
                return jsonify({"messages": out})
            expr = sp.sympify(user_input1, locals=locals_dict)
            intervalle = _parse_interval_payload(data)
            try:
                N_harm = int(user_input2) if user_input2 else 10
            except Exception:
                N_harm = 10
            if N_harm < 1:
                N_harm = 10
            _fourier.etude_serie_de_fourier(expr, x, intervalle, N_harm)
            return jsonify({"messages": out})

        if type_input == "Intégrale de Lebesgue":
            # Étude d'une intégrale (propre / impropre / généralisée, L1) : module lebesgue.py
            try:
                import lebesgue as _leb
            except Exception as _e:
                st.latex(_latex_rouge("Le module lebesgue.py est introuvable : placez-le à côté de app.py."))
                return jsonify({"messages": out})
            _leb.st = st
            x = sp.symbols("x", real=True)
            locals_dict["x"] = x
            if not user_input1:
                st.latex(_latex_rouge("Veuillez entrer la fonction f(x) à intégrer."))
                return jsonify({"messages": out})
            expr = sp.sympify(user_input1, locals=locals_dict)
            intervalle = _parse_interval_payload(data)
            _leb.etude_integrale(expr, x, intervalle, 10)
            return jsonify({"messages": out})

        if type_input == "Théorèmes de convergence (intégrales)":
            # Convergence monotone / dominée / Fatou pour une suite d'intégrales : module lebesgue.py
            try:
                import lebesgue as _leb
            except Exception as _e:
                st.latex(_latex_rouge("Le module lebesgue.py est introuvable : placez-le à côté de app.py."))
                return jsonify({"messages": out})
            _leb.st = st
            _leb.cv_limite = _cv_limite_ponctuelle
            _leb.cv_fusionne = _cv_fusionne
            x = sp.symbols("x", real=True)
            n = sp.symbols("n", integer=True, nonnegative=True)
            locals_dict["x"] = x
            locals_dict["n"] = n
            if not user_input1:
                st.latex(_latex_rouge("Veuillez entrer l'expression de f_n(x) (en x et n)."))
                return jsonify({"messages": out})
            expr = sp.sympify(user_input1, locals=locals_dict)
            intervalle = _parse_interval_payload(data)
            _leb.etude_theoremes_convergence(expr, n, x, intervalle, 10)
            return jsonify({"messages": out})

        if type_input == "Intégrale multiple":
            # Intégrales multiples (Fubini / Fubini-Tonelli, changements de variables) : module multi.py
            try:
                import lebesgue as _leb
                import multi as _multi
            except Exception as _e:
                st.latex(_latex_rouge("Les modules lebesgue.py et multi.py sont introuvables : placez-les à côté de app.py."))
                return jsonify({"messages": out})
            _leb.st = st
            _multi.st = st
            if not user_input1:
                st.latex(_latex_rouge("Veuillez entrer la fonction f à intégrer."))
                return jsonify({"messages": out})
            _multi.etude_integrale_multiple(user_input1, (data.get("user_input3") or "").strip(), (data.get("user_input4") or "").strip(), locals_dict)
            return jsonify({"messages": out})

        if type_input == "Intégrale curviligne / Green–Riemann":
            # Intégrales curvilignes, champs de gradient, Green–Riemann : module curv.py
            try:
                import lebesgue as _leb
                import curv as _curv
            except Exception as _e:
                st.latex(_latex_rouge("Les modules lebesgue.py et curv.py sont introuvables : placez-les à côté de app.py."))
                return jsonify({"messages": out})
            _leb.st = st
            _curv.st = st
            _curv.etude_curviligne(user_input1, (data.get("user_input3") or "").strip(), locals_dict)
            return jsonify({"messages": out})

        # Cas Suite/Série numériques
        if not user_input1:
            st.latex(_latex_rouge("Veuillez entrer une expression correcte."))
            return jsonify({"messages": out})

        u = sp.sympify(user_input1, locals=locals_dict)
        u_n = sp.simplify(u)

        if type_input == "Série":
            if not user_input2:
                st.latex(_latex_rouge("Veuillez renseigner l'indice de départ."))
                return jsonify({"messages": out})

            n_min = int(user_input2)
            kk = sp.symbols("k", integer=True)
            S_n = sp.Sum(u_n.subs(n, kk), (kk, n_min, n))

            st.latex(r"\text{Vous avez entré la série de somme partielle :}")
            st.latex(sp.latex(S_n))
            st.latex(r"\text{Son terme général est : } u_n=" + sp.latex(u_n))

            try:
                critere(u_n, n, n_min)  # 
            except Exception:
                pass

            try:
                graphe(S_n, n, "blue") 
                graphe(u_n, n, "red")   
            except Exception:
                pass

            return jsonify({"messages": out})

        # Suite numérique
        st.latex(r"\text{Vous avez entré la suite définie par }u_n=" + sp.latex(u_n))
        lim = sp.limit_seq(u_n, n)
        st.latex(r"\lim_{n\to\infty} " + sp.latex(u_n) + "=" + sp.latex(lim))

        if getattr(lim, "is_real", False):
            st.latex(r"\text{La suite admet une limite réelle : convergence.}")
        else:
            st.latex(r"\text{La suite ne possède pas de limite réelle : divergence.}")

        try:
            graphe(u_n, n, "blue")  # noqa: F821
        except Exception:
            pass

        return jsonify({"messages": out})

    except Exception as e:
        st.latex(_latex_rouge("Entrée invalide. Veuillez entrer une expression correcte."))
        st.latex(r"\text{Détail technique : " + str(e).replace("_", r"\_") + r"}")
        return jsonify({"messages": out})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)