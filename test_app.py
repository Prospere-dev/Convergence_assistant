"""Tests de base. Lancer depuis le dossier du projet :
    .venv\\Scripts\\python -m pip install pytest
    .venv\\Scripts\\python -m pytest -v
"""
import pytest
import sympy as sp

import app as A


@pytest.fixture
def client():
    A.app.config["TESTING"] = True
    return A.app.test_client()


def compute(client, **payload):
    payload.setdefault("user_input2", "1")
    payload.setdefault("interval_left", "")
    payload.setdefault("interval_right", "")
    r = client.post("/compute", json=payload)
    assert r.status_code == 200
    return r.get_json()["messages"]


def texte(messages):
    return " ".join(m["content"] for m in messages if m["type"] != "plot")


# --- Bugs repérés à la lecture du code ---------------------------------
def test_facteur2_ne_plante_pas():
    n = sp.symbols("n", integer=True, nonnegative=True)
    A.facteur2(2 * sp.pi * n)  # UnboundLocalError attendu avant correction


def test_coth_est_bien_coth():
    # coth(1) = 1.3130...  (acoth(1) = oo)
    assert abs(float(sp.coth(1)) - 1.3130352855) < 1e-9
    # Vérifie la table du dictionnaire : à adapter si tu l'extrais en constante.
    src = open(A.__file__, encoding="utf-8").read()
    assert '"coth": sp.coth' in src


# --- Fumée : aucune erreur serveur sur des cas classiques --------------
@pytest.mark.parametrize("expr", ["1/n**2", "1/n", "(-1)**n/n", "n/(n+1)"])
def test_series_pas_d_erreur(client, expr):
    msgs = compute(client, type_input="Série", user_input1=expr, user_input2="1")
    assert msgs
    assert not any(m["type"] == "error" for m in msgs)
    assert "Entrée invalide" not in texte(msgs)


def test_suite_limite(client):
    msgs = compute(client, type_input="Suite", user_input1="n/(n+1)")
    assert "convergence" in texte(msgs)


def test_suite_divergente(client):
    msgs = compute(client, type_input="Suite", user_input1="n")
    assert "divergence" in texte(msgs)


def test_serie_entiere_exp(client):
    msgs = compute(client, type_input="Série entière", user_input1="x**n/factorial(n)", user_input2="0")
    assert not any(m["type"] == "error" for m in msgs)
