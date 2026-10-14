/* ===================================================================================
   app.js
   Interface uniquement : aucune logique mathématique
   Communication avec Flask + gestion de l’affichage
   =================================================================================== */

/* ------------------------------------------------
   Affichage conditionnel des champs selon le type
------------------------------------------------ */

function updateFormByType() {
  const type = document.getElementById("type_input").value;

  const blockInterval = document.getElementById("interval_block");
  const labelInput2 = document.getElementById("label_input2");
  const input2 = document.getElementById("user_input2");
  const labelInput1 = document.getElementById("label_input1");
  const input1 = document.getElementById("user_input1");

  const lab3 = document.getElementById("label_input3");
  const in3 = document.getElementById("user_input3");
  const lab4 = document.getElementById("label_input4");
  const in4 = document.getElementById("user_input4");
  if (!in3.dataset.ph) { in3.dataset.ph = in3.placeholder; lab3.dataset.tx = lab3.innerText; }
  in3.placeholder = in3.dataset.ph; lab3.innerText = lab3.dataset.tx;
  lab3.style.display = "none"; in3.style.display = "none";
  lab4.style.display = "none"; in4.style.display = "none";

  // Par défaut
  blockInterval.style.display = "none";
  labelInput2.style.display = "none";
  input2.style.display = "none";
  labelInput2.innerText = "Indice minimal (si nécessaire) :";

  // Ajustement des labels selon le type
  if (type === "Suite") {
    labelInput1.innerText = "Terme général u_n :";
    input1.placeholder = "Ex : 1/n, (-1)^n/n, n/(n+1)";
  }

  if (type === "Série") {
    labelInput1.innerText = "Terme général u_n :";
    input1.placeholder = "Ex : 1/n^2, (-1)^n/n";
    labelInput2.style.display = "block";
    input2.style.display = "block";
  }

  if (type === "Suites de fonctions") {
    labelInput1.innerText = "Terme général f_n(x) :";
    input1.placeholder = "Ex : sin(n*x)/n, x^n/(1+n)";
    blockInterval.style.display = "block";
  }

  if (type === "Séries de fonctions") {
    labelInput1.innerText = "Terme général u_n(x) :";
    input1.placeholder = "Ex : x^n/n^2, sin(n*x)/n";
    labelInput2.style.display = "block";
    input2.style.display = "block";
    blockInterval.style.display = "block";
  }

  if (type === "Série entière") {
    labelInput1.innerText = "Terme général a_n x^n :";
    input1.placeholder = "Ex : x^n/n!, (-1)^n*x^n, (x-1)^n";
    labelInput2.style.display = "block";
    input2.style.display = "block";
    blockInterval.style.display = "block";
  }

  if (type === "Série de Fourier") {
    labelInput1.innerText = "Fonction f(x), périodique, donnée sur une période :";
    input1.placeholder = "Ex : x, abs(x), x^2, Piecewise((-1, x<0), (1, True))";
    labelInput2.innerText = "Nombre d'harmoniques N (pour les graphes) :";
    labelInput2.style.display = "block";
    input2.style.display = "block";
    blockInterval.style.display = "block";
  }

  if (type === "Intégrale de Lebesgue") {
    labelInput1.innerText = "Fonction f(x) à intégrer :";
    input1.placeholder = "Ex : 1/(1+x^2), sin(x)/x, 1/sqrt(x), exp(-x^2)";
    blockInterval.style.display = "block";
  }

  if (type === "Théorèmes de convergence (intégrales)") {
    labelInput1.innerText = "Suite de fonctions f_n(x) (en x et n) :";
    input1.placeholder = "Ex : n*x*exp(-n*x^2), x^n, exp(-n*x)/(1+x^2), sin(n*x)/(n*(1+x^2))";
    blockInterval.style.display = "block";
  }

  if (type === "Intégrale multiple") {
    labelInput1.innerText = "Fonction f des variables (x, y, z, ... au choix) :";
    input1.placeholder = "Ex : x*y, exp(-x^2-y^2), 1/(x^2+y^2), x1*x2*x3*x4";
    lab3.style.display = "block"; in3.style.display = "block";
    lab4.style.display = "block"; in4.style.display = "block";
  }

  if (type === "Intégrale curviligne / Green–Riemann") {
    labelInput1.innerText = "Champ V = (P, Q[, R]), fonction f (intégrale ∫f ds), ou « longueur » / « aire » :";
    input1.placeholder = "Ex : -y, x  |  x^2+y, x*y  |  y*z, x*z, x*y  |  x*y  |  longueur  |  aire";
    lab3.innerText = "Courbe γ (arc paramétré) ou domaine D (Green–Riemann) :";
    in3.placeholder = "Arc : x=cos(t), y=sin(t), t in [0,2*pi]   |   Domaine : x^2+y^2<=1 ; 0<=x<=1, x^2<=y<=x";
    lab3.style.display = "block"; in3.style.display = "block";
  }

  if (type === "Développement en série entière") {
    labelInput1.innerText = "Fonction f(x) :";
    input1.placeholder = "Ex : exp(x), 1/(1-x), ln(1+x)";
    blockInterval.style.display = "block";
  }
}

/* ------------------------------------------------
   Envoi des données au serveur Flask
------------------------------------------------ */

function compute() {
  const payload = {};

  // Type d’étude
  payload.type_input = document.getElementById("type_input").value;

  // Entrées principales
  payload.user_input1 = document.getElementById("user_input1").value.trim();
  payload.user_input2 = document.getElementById("user_input2").value.trim();
  payload.user_input3 = (document.getElementById("user_input3")?.value ?? "").trim();
  payload.user_input4 = (document.getElementById("user_input4")?.value ?? "").trim();

  /* ------------------------------------------------
     INTERVALLE
     Convention :
     Borne gauche :
       "("  → ouvert  → closed = false
       "["  → fermé   → closed = true
       "]"  → ouvert (notation française) → closed = false
     Borne droite :
       ")"  → ouvert  → closed = false
       "]"  → fermé   → closed = true
       "["  → ouvert (notation française) → closed = false
  ------------------------------------------------ */

  const leftBracket  = document.getElementById("left_bracket")?.value ?? "(";
  const rightBracket = document.getElementById("right_bracket")?.value ?? ")";

  const intervalLeft  = document.getElementById("interval_left")?.value.trim() ?? "";
  const intervalRight = document.getElementById("interval_right")?.value.trim() ?? "";

  payload.interval_left_closed  = (leftBracket === "[");
  payload.interval_right_closed = (rightBracket === "]");

  payload.interval_left  = intervalLeft;
  payload.interval_right = intervalRight;

  // Envoi au serveur
  fetch("/compute", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  })
  .then(response => response.json())
  .then(data => afficherResultats(data))
  .catch(err => {
    console.error("Erreur serveur :", err);
  });
}

/* ------------------------------------------------
   Affichage des résultats renvoyés par Flask
------------------------------------------------ */

function afficherResultats(data) {
  const zone = document.getElementById("resultats");
  zone.innerHTML = "";

  (data.messages || []).forEach(msg => {
    const div = document.createElement("div");
    div.className = "msg";

    if (msg.type === "latex") {
      div.classList.add("latex");
      div.innerHTML = "$$" + msg.content + "$$";
    }
    else if (msg.type === "info") {
      div.classList.add("info");
      div.innerText = msg.content;
    }
    else if (msg.type === "error") {
      div.classList.add("error");
      div.innerText = msg.content;
    }
    else if (msg.type === "plot") {
      const img = document.createElement("img");
      img.src = "data:image/png;base64," + msg.content;
      img.className = "plot";
      div.appendChild(img);
    }

    zone.appendChild(div);
  });

  // Rafraîchissement MathJax
  if (window.MathJax) {
    MathJax.typesetPromise();
  }
}

/* ------------------------------------------------
   Initialisation au chargement de la page
------------------------------------------------ */

document.addEventListener("DOMContentLoaded", () => {
  // 1) Affichage / masquage du bloc intervalle dès le chargement
  updateFormByType();

  // 2) Mise à jour automatique quand on change le type d’étude
  document.getElementById("type_input")
          .addEventListener("change", updateFormByType);

  // 3) Bouton de lancement
  document.getElementById("btn_run")
          .addEventListener("click", compute);
});