# Projet ML — Prédiction d'AVC (`Projet_ML_AVC.ipynb`)

Ce dépôt contient un notebook Jupyter dédié à la **prédiction du risque d'AVC (Accident Vasculaire Cérébral)** à partir de données tabulaires. Le notebook couvre l’ensemble du flux classique d’un projet de machine learning : exploration des données, préparation, sélection de modèles, entraînement, évaluation et interprétation.

---

##  Objectifs
- Charger et analyser un jeu de données lié aux facteurs de risque d’AVC.
- Nettoyer, préparer et vectoriser les variables (numériques et catégorielles).
- Entraîner et comparer plusieurs modèles de ML (baseline vs. modèles plus avancés).
- Évaluer les performances (précision, rappel, F1, AUC, matrice de confusion).
- Documenter les résultats et proposer des pistes d’amélioration.
- (Optionnel) Expliquer les prédictions avec des méthodes d’interprétabilité (permutations, SHAP, etc.).

---

##  Contenu
- `Projet_ML_AVC.ipynb` — Notebook principal.
- *(Optionnel)* `data/` — Dossier local pour les données (non versionnées).
- *(Optionnel)* `models/` — Dossier de sauvegarde des modèles et artefacts.
- *(Optionnel)* `reports/` — Graphiques, figures et tableaux de résultats exportés.

> Si vous travaillez dans un environnement où le suivi Git est actif, pensez à ajouter un `.gitignore` pour exclure `data/`, `models/` et les fichiers temporaires.

---

##  Installation & environnement

### 1) Créer un environnement Python
Vous pouvez utiliser **venv** ou **conda**.

**Avec `venv`:**
```bash
python -m venv .venv
# Activer l'environnement
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux (bash/zsh)
source .venv/bin/activate
```

**Avec `conda`:**
```bash
conda create -n ml-avc python=3.10 -y
conda activate ml-avc
```

### 2) Installer les dépendances
Si un `requirements.txt` n’est pas fourni, installez les bibliothèques usuelles du workflow ML :
```bash
pip install numpy pandas scikit-learn matplotlib plotly seaborn jupyter ipykernel
# (Optionnel) interprétabilité & gestion des déséquilibres
pip install shap imbalanced-learn
```

> Ajoutez votre environnement au noyau Jupyter si besoin :
```bash
python -m ipykernel install --user --name ml-avc --display-name "Python (ml-avc)"
```

---

##  Données

Placez le jeu de données dans `data/` ou adaptez le chemin dans le notebook.  
Exemples de colonnes souvent rencontrées (à adapter) : `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, `stroke`.

> **Imbalance** : Le label `stroke` est généralement **très déséquilibré**. Pensez à :
- pondérer les classes (`class_weight="balanced"`),
- utiliser du *resampling* (SMOTE, RandomUnderSampler),
- choisir des métriques robustes (AUC, rappel classe minoritaire, F1-positif).

---

##  Exécution

### Option A — Jupyter Lab/Notebook
```bash
jupyter lab
# ou
jupyter notebook
```
Ouvrez `Projet_ML_AVC.ipynb` et exécutez les cellules dans l’ordre.

### Option B — VS Code
- Installez l’extension **Python** et **Jupyter**.
- Sélectionnez le noyau correspondant à votre environnement (en haut à droite du notebook).
- Exécutez cellule par cellule.

---

##  Métriques & rapports
Le notebook calcule notamment :
- **AUC** ROC / PR
- **Précision / Rappel / F1** (macro, weighted)
- **Matrice de confusion**
- **Courbes ROC/PR** et importances des variables (selon le modèle)

> Pour problèmes déséquilibrés : privilégiez **rappel** de la classe positive et **AUC-PR**.

---

##  Reproductibilité
- Fixez des graines aléatoires (`random_state`) pour `train_test_split`, `StratifiedKFold`, et les modèles.
- Versionnez vos notebooks nettoyés (évitez les sorties volumineuses).
- Sauvegardez la configuration logicielle :
```bash
pip freeze > requirements.txt
# ou
conda env export > environment.yml
```

---

##  Structure de pipeline (référence)
- **Prétraitement** : imputations, encodage (One-Hot/Ordinal), normalisation/standardisation.
- **Validation** : `StratifiedKFold` ou `StratifiedShuffleSplit`.
- **Modèles** : Logistic Regression (baseline), Random Forest / XGBoost / LightGBM (si dispo), SVM, etc.
- **Recherche d’hyperparamètres** : `GridSearchCV` / `RandomizedSearchCV`.
- **Interprétabilité** : SHAP, importance permutation, PDP/ICE.

---

##  Éthique & biais
- Vérifiez les biais potentiels (genre, âge, statut socio-économique).
- Documentez les limites et usages responsables : ce modèle **n’est pas un dispositif médical**.
- Évitez l’usage en décision automatisée sans validation clinique et revue éthique.

---

##  Pistes d’amélioration
- Collecte de données supplémentaires et meilleure qualité (bmi manquants, variables cliniques).
- Ingénierie de variables (interactions, transformations non linéaires).
- Seuils de décision adaptés au **coût d’erreur** (favoriser le rappel).
- Calibration des probabilités (Platt/Isotonic).
- *Ensembles* de modèles et *stacking*.
- Explicabilité avancée et tableaux de bord (Dash/Streamlit).

---

##  Références (génériques)
- Scikit-learn (User Guide) — https://scikit-learn.org/stable/
- Imbalanced-learn — https://imbalanced-learn.org/stable/
- SHAP — https://shap.readthedocs.io/

---


##  Auteur
Nassim LOUDIYI / Paul-Adrien LU-YEN-TUNG


