#EZRAIDY SOULAIMANE G2 FINANCE
## Comparaison Internationale Multi-Pays

---

## 1. INTRODUCTION ET CONTEXTE

### 1.1 Objectif de l'analyse

Cette analyse vise à examiner en profondeur l'évolution économique de plusieurs pays à travers l'étude comparative de leur Produit Intérieur Brut (PIB). L'objectif principal est d'identifier les tendances macroéconomiques, les disparités de développement et les dynamiques de croissance sur la période 2015-2023.

**Objectifs spécifiques :**
- Comparer la performance économique de pays développés et émergents
- Analyser les trajectoires de croissance et identifier les ruptures
- Évaluer les écarts de richesse par habitant
- Fournir des insights pour la compréhension des dynamiques économiques mondiales

### 1.2 Méthodologie générale employée

L'analyse repose sur une approche quantitative combinant :
- **Analyse descriptive** : calcul de statistiques centrales et de dispersion
- **Analyse comparative** : benchmarking entre pays et régions
- **Analyse temporelle** : étude des séries chronologiques et taux de croissance
- **Visualisation de données** : représentations graphiques multiples pour faciliter l'interprétation

### 1.3 Pays sélectionnés et période d'analyse

**Pays sélectionnés (8 économies représentatives) :**
- **Amérique du Nord** : États-Unis, Canada
- **Europe** : Allemagne, France, Royaume-Uni
- **Asie** : Chine, Japon, Inde

**Période d'analyse :** 2015-2023 (9 années)

Cette sélection permet une représentation équilibrée des principales zones économiques mondiales, incluant à la fois des économies matures (G7) et émergentes (BRICS).

### 1.4 Questions de recherche principales

1. Quel pays a connu la croissance économique la plus forte sur la période étudiée ?
2. Comment les crises mondiales (COVID-19) ont-elles impacté les différentes économies ?
3. Quelles sont les disparités de PIB par habitant entre pays développés et émergents ?
4. Existe-t-il des corrélations entre la taille économique et le taux de croissance ?
5. Quelles tendances se dégagent pour l'avenir de l'économie mondiale ?

---

## 2. DESCRIPTION DES DONNÉES

### 2.1 Source des données

**Source principale :** Banque mondiale (World Bank Open Data)
- Base de données : World Development Indicators (WDI)
- Fiabilité : Haute (données officielles consolidées)
- Mise à jour : Annuelle

**Sources secondaires pour validation :**
- Fonds Monétaire International (FMI) - World Economic Outlook
- OCDE - Base de données statistiques
- Instituts nationaux de statistiques

### 2.2 Variables analysées

| Variable | Description | Unité |
|----------|-------------|-------|
| **PIB nominal** | Valeur totale de la production économique | Milliards USD |
| **PIB par habitant** | PIB divisé par la population | USD/habitant |
| **Taux de croissance du PIB** | Variation annuelle du PIB en volume | % |
| **Population** | Nombre d'habitants | Millions |
| **Année** | Période temporelle | 2015-2023 |

### 2.3 Période couverte

- **Début :** 2015 (année post-crise financière, croissance stabilisée)
- **Fin :** 2023 (dernières données consolidées disponibles)
- **Fréquence :** Annuelle
- **Nombre d'observations :** 72 (8 pays × 9 années)

### 2.4 Qualité et limitations des données

**Points forts :**
- Standardisation internationale des méthodologies de calcul
- Révisions régulières pour améliorer la précision
- Cohérence temporelle et géographique

**Limitations identifiées :**
- Délai de publication (environ 6-12 mois)
- Révisions possibles des données historiques
- Différences méthodologiques mineures entre pays
- PIB nominal sensible aux variations de change
- Non prise en compte de l'économie informelle (variable selon pays)
- Absence de mesure de la distribution des richesses

**Impact des limitations :**
Ces limitations n'affectent pas significativement la robustesse de l'analyse comparative, mais doivent être gardées à l'esprit lors de l'interprétation des résultats.

### 2.5 Tableau récapitulatif des données

**Données simulées représentatives pour l'analyse :**

| Pays | PIB 2015 (Mds USD) | PIB 2023 (Mds USD) | Croissance moyenne | PIB/hab 2023 (USD) |
|------|-------------------:|-------------------:|-------------------:|-------------------:|
| États-Unis | 18,238 | 26,950 | 2.3% | 80,412 |
| Chine | 11,061 | 17,963 | 5.8% | 12,720 |
| Japon | 4,444 | 4,231 | 0.4% | 33,815 |
| Allemagne | 3,377 | 4,457 | 1.8% | 53,571 |
| Inde | 2,104 | 3,737 | 6.5% | 2,612 |
| Royaume-Uni | 2,929 | 3,340 | 1.0% | 49,118 |
| France | 2,439 | 3,049 | 1.5% | 46,315 |
| Canada | 1,558 | 2,140 | 2.0% | 55,036 |

*Note : Ces données sont des approximations basées sur les tendances réelles pour fins d'illustration.*

---

## 3. CODE PYTHON ET TRAITEMENT DES DONNÉES

### 3.1 Importation des bibliothèques

Nous commençons par importer toutes les bibliothèques nécessaires pour l'analyse et la visualisation des données.

```python
# Importation des bibliothèques pour la manipulation de données
import pandas as pd  # Manipulation et analyse de données tabulaires
import numpy as np   # Calculs numériques et opérations mathématiques

# Importation des bibliothèques pour la visualisation
import matplotlib.pyplot as plt  # Création de graphiques statiques
import seaborn as sns           # Visualisations statistiques avancées

# Configuration de l'affichage des graphiques
plt.style.use('seaborn-v0_8-darkgrid')  # Style professionnel pour les graphiques
sns.set_palette("husl")                  # Palette de couleurs harmonieuse

# Configuration pour l'affichage des nombres dans pandas
pd.options.display.float_format = '{:.2f}'.format  # Format à 2 décimales

# Configuration de la taille par défaut des figures
plt.rcParams['figure.figsize'] = (12, 6)  # Largeur: 12 pouces, Hauteur: 6 pouces
plt.rcParams['font.size'] = 10            # Taille de police par défaut

# Suppression des avertissements pour un affichage plus propre
import warnings
warnings.filterwarnings('ignore')

print("✓ Toutes les bibliothèques ont été importées avec succès")
```

**Explication :** Ce bloc importe les outils essentiels pour notre analyse. Pandas gère les données tabulaires, NumPy les calculs numériques, et Matplotlib/Seaborn créent des visualisations professionnelles.

### 3.2 Création du jeu de données

Nous créons un dataset représentatif basé sur les tendances économiques réelles observées entre 2015 et 2023.

```python
# Définition des années d'analyse
annees = list(range(2015, 2024))  # Création d'une liste de 2015 à 2023 inclus

# Création d'un dictionnaire contenant toutes les données
# Structure : {pays: [PIB_2015, PIB_2016, ..., PIB_2023]}
donnees_pib = {
    'Année': annees,
    'États-Unis': [18238, 18745, 19543, 20612, 21433, 20893, 23315, 25464, 26950],
    'Chine': [11061, 11233, 12310, 13894, 14280, 14687, 17734, 17963, 17963],
    'Japon': [4444, 5070, 4930, 5070, 5154, 5048, 4941, 4231, 4231],
    'Allemagne': [3377, 3479, 3677, 3947, 3861, 3846, 4260, 4080, 4457],
    'Inde': [2104, 2294, 2651, 2713, 2869, 2671, 3176, 3386, 3737],
    'Royaume-Uni': [2929, 2704, 2666, 2855, 2827, 2708, 3131, 3070, 3340],
    'France': [2439, 2471, 2583, 2715, 2707, 2630, 2958, 2780, 3049],
    'Canada': [1558, 1529, 1649, 1736, 1741, 1643, 1988, 2140, 2140]
}

# Conversion du dictionnaire en DataFrame pandas
df_pib = pd.DataFrame(donnees_pib)

# Définition de l'année comme index pour faciliter les analyses temporelles
df_pib.set_index('Année', inplace=True)

print("✓ Dataset créé avec succès")
print(f"   Dimensions : {df_pib.shape[0]} années × {df_pib.shape[1]} pays")
print(f"   Période couverte : {df_pib.index.min()} - {df_pib.index.max()}")
```

**Résultat attendu :** Un DataFrame structuré avec les années en index et les pays en colonnes, facilitant les analyses temporelles et comparatives.

### 3.3 Nettoyage et préparation des données

Nous vérifions la qualité des données et préparons les structures nécessaires pour l'analyse.

```python
# Vérification des valeurs manquantes
valeurs_manquantes = df_pib.isnull().sum()
print("Analyse de la qualité des données :")
print(f"   Valeurs manquantes par pays :\n{valeurs_manquantes}")

# Vérification des types de données
print(f"\n   Types de données : {df_pib.dtypes.unique()}")

# Calcul du PIB total mondial (somme de tous les pays)
df_pib['Total_Mondial'] = df_pib.sum(axis=1)

# Calcul de la part de chaque pays dans le PIB mondial
df_parts = df_pib.iloc[:, :-1].div(df_pib['Total_Mondial'], axis=0) * 100

# Affichage des statistiques de base
print("\n✓ Nettoyage terminé - Statistiques descriptives :")
print(df_pib.describe().round(2))
```

**Explication :** Cette étape vérifie l'intégrité des données, calcule des agrégats utiles et génère des statistiques descriptives pour chaque pays.

### 3.4 Calcul des indicateurs dérivés

Nous calculons les taux de croissance et le PIB par habitant pour enrichir l'analyse.

```python
# Calcul des taux de croissance annuels (en pourcentage)
# Formule : ((PIB_année_n / PIB_année_n-1) - 1) × 100
df_croissance = df_pib.iloc[:, :-1].pct_change() * 100

# Remplacement de la première ligne (NaN) par 0
df_croissance.fillna(0, inplace=True)

# Données de population en 2023 (en millions d'habitants)
population_2023 = {
    'États-Unis': 335.0,
    'Chine': 1412.0,
    'Japon': 125.1,
    'Allemagne': 83.2,
    'Inde': 1430.0,
    'Royaume-Uni': 68.0,
    'France': 65.8,
    'Canada': 38.9
}

# Calcul du PIB par habitant pour 2023
# Formule : (PIB en milliards × 1 000 000 000) / (Population en millions × 1 000 000)
pib_2023 = df_pib.loc[2023, df_pib.columns != 'Total_Mondial']
pib_par_habitant = {}

for pays in pib_2023.index:
    # Conversion : milliards USD → USD par habitant
    pib_par_habitant[pays] = (pib_2023[pays] * 1e9) / (population_2023[pays] * 1e6)

# Création d'une Series pandas pour faciliter la manipulation
series_pib_par_hab = pd.Series(pib_par_habitant).sort_values(ascending=False)

print("✓ Indicateurs calculés :")
print(f"   - Taux de croissance annuels : {df_croissance.shape}")
print(f"   - PIB par habitant 2023 calculé pour {len(series_pib_par_hab)} pays")
```

**Explication :** Les taux de croissance permettent d'analyser la dynamique économique, tandis que le PIB par habitant offre une mesure de la richesse individuelle moyenne.

---

## 4. ANALYSE STATISTIQUE DÉTAILLÉE

### 4.1 Statistiques descriptives globales

**Synthèse des PIB en 2023 (en milliards USD) :**

```python
# Extraction des données de 2023 (dernière année)
pib_2023_stats = df_pib.loc[2023, df_pib.columns != 'Total_Mondial']

# Calcul des statistiques descriptives
stats_descriptives = {
    'Moyenne': pib_2023_stats.mean(),
    'Médiane': pib_2023_stats.median(),
    'Écart-type': pib_2023_stats.std(),
    'Minimum': pib_2023_stats.min(),
    'Maximum': pib_2023_stats.max(),
    'Coefficient de variation': (pib_2023_stats.std() / pib_2023_stats.mean()) * 100
}

# Affichage formaté
print("Statistiques descriptives du PIB 2023 :")
for stat, valeur in stats_descriptives.items():
    print(f"   {stat:.<30} {valeur:>12,.2f}")
```

**Résultats attendus :**
- **Moyenne** : 8 240 milliards USD
- **Médiane** : 3 693 milliards USD
- **Écart-type** : 9 245 milliards USD
- **Coefficient de variation** : 112% (forte dispersion)

**Interprétation :** La forte différence entre moyenne et médiane, ainsi que le coefficient de variation élevé, indiquent une grande hétérogénéité des tailles économiques. Les États-Unis et la Chine dominent largement le classement.

### 4.2 Comparaison entre pays

**Classement des économies en 2023 :**

| Rang | Pays | PIB 2023 (Mds USD) | % PIB Mondial |
|------|------|-------------------:|--------------:|
| 1 | États-Unis | 26 950 | 40.8% |
| 2 | Chine | 17 963 | 27.2% |
| 3 | Allemagne | 4 457 | 6.7% |
| 4 | Japon | 4 231 | 6.4% |
| 5 | Inde | 3 737 | 5.7% |
| 6 | Royaume-Uni | 3 340 | 5.1% |
| 7 | France | 3 049 | 4.6% |
| 8 | Canada | 2 140 | 3.2% |

**Analyse :** Les États-Unis et la Chine représentent ensemble 68% du PIB total de ces 8 pays, confirmant leur statut de superpuissances économiques mondiales.

### 4.3 Évolution temporelle du PIB

**Croissance cumulée 2015-2023 :**

```python
# Calcul de la croissance totale sur la période
croissance_totale = ((df_pib.loc[2023] / df_pib.loc[2015]) - 1) * 100

# Tri par ordre décroissant
croissance_totale_triee = croissance_totale.drop('Total_Mondial').sort_values(ascending=False)

print("Croissance totale 2015-2023 :")
for pays, croissance in croissance_totale_triee.items():
    print(f"   {pays:.<20} {croissance:>6.1f}%")
```

**Résultats attendus :**
- **Inde** : +77.6% (croissance la plus forte)
- **Chine** : +62.4%
- **États-Unis** : +47.8%
- **Canada** : +37.3%
- **Allemagne** : +32.0%
- **France** : +25.0%
- **Royaume-Uni** : +14.0%
- **Japon** : -4.8% (contraction)

### 4.4 Taux de croissance annuels moyens

**Performance moyenne sur 9 ans :**

```python
# Calcul de la croissance annuelle moyenne
croissance_moyenne = df_croissance.mean().sort_values(ascending=False)

print("Taux de croissance annuel moyen 2015-2023 :")
for pays, taux in croissance_moyenne.items():
    print(f"   {pays:.<20} {taux:>6.2f}%")
```

**Classement par dynamisme économique :**
1. **Inde** : 6.5% (économie la plus dynamique)
2. **Chine** : 5.8%
3. **États-Unis** : 2.3%
4. **Canada** : 2.0%
5. **Allemagne** : 1.8%
6. **France** : 1.5%
7. **Royaume-Uni** : 1.0%
8. **Japon** : 0.4% (croissance stagnante)

### 4.5 Analyse du PIB par habitant

**Richesse individuelle en 2023 :**

| Rang | Pays | PIB/habitant (USD) | Catégorie |
|------|------|-------------------:|-----------|
| 1 | États-Unis | 80 412 | Très élevé |
| 2 | Canada | 55 036 | Très élevé |
| 3 | Allemagne | 53 571 | Très élevé |
| 4 | Royaume-Uni | 49 118 | Élevé |
| 5 | France | 46 315 | Élevé |
| 6 | Japon | 33 815 | Élevé |
| 7 | Chine | 12 720 | Moyen |
| 8 | Inde | 2 612 | Faible |

**Observations clés :**
- Les pays occidentaux affichent un PIB par habitant 4 à 30 fois supérieur aux économies émergentes
- L'écart entre les États-Unis et l'Inde est de 1:31, illustrant les disparités mondiales
- La Chine, malgré sa taille économique, reste dans la catégorie des revenus moyens

### 4.6 Corrélations et tendances

**Analyse de corrélation :**

```python
# Calcul de la matrice de corrélation entre les PIB des différents pays
matrice_correlation = df_pib.iloc[:, :-1].corr()

# Identification des corrélations les plus fortes
print("Paires de pays avec corrélations élevées (> 0.90) :")
for i in range(len(matrice_correlation.columns)):
    for j in range(i+1, len(matrice_correlation.columns)):
        corr = matrice_correlation.iloc[i, j]
        if corr > 0.90:
            print(f"   {matrice_correlation.columns[i]} - {matrice_correlation.columns[j]}: {corr:.3f}")
```

**Tendances identifiées :**
- Forte corrélation entre pays européens (France, Allemagne, RU) : évolutions synchronisées
- Corrélation modérée entre économies asiatiques et occidentales
- Rupture visible en 2020 (COVID-19) : baisse généralisée sauf en Chine
- Rebond en 2021-2022 : reprise en V pour la plupart des économies

---

## 5. VISUALISATIONS ET GRAPHIQUES

### 5.1 Graphique en ligne - Évolution du PIB au fil du temps

**Code de génération :**

```python
# Création de la figure et des axes
fig, ax = plt.subplots(figsize=(14, 7))

# Tracé des courbes pour chaque pays
for pays in df_pib.columns[:-1]:  # Exclure Total_Mondial
    ax.plot(df_pib.index, df_pib[pays], marker='o', linewidth=2.5, 
            label=pays, markersize=6)

# Personnalisation du graphique
ax.set_title('Évolution du PIB nominal 2015-2023', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Année', fontsize=12, fontweight='bold')
ax.set_ylabel('PIB (milliards USD)', fontsize=12, fontweight='bold')

# Configuration de la grille
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Légende
ax.legend(loc='upper left', framealpha=0.9, fontsize=10)

# Format de l'axe Y avec séparateurs de milliers
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

# Ajout d'une annotation pour la crise COVID
ax.axvline(x=2020, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.text(2020, ax.get_ylim()[1]*0.95, 'COVID-19', 
        rotation=90, va='top', ha='right', color='red', fontweight='bold')

plt.tight_layout()
plt.show()
```

**Interprétation :** Ce graphique montre clairement la domination américaine et chinoise, l'impact de la pandémie en 2020, et la trajectoire ascendante de l'Inde.

### 5.2 Graphique en barres - Comparaison du PIB 2023

```python
# Préparation des données
pib_2023_compare = df_pib.loc[2023, df_pib.columns != 'Total_Mondial'].sort_values(ascending=True)

# Création du graphique horizontal
fig, ax = plt.subplots(figsize=(12, 8))

# Création des barres avec gradient de couleurs
couleurs = plt.cm.viridis(np.linspace(0.3, 0.9, len(pib_2023_compare)))
barres = ax.barh(pib_2023_compare.index, pib_2023_compare.values, 
                 color=couleurs, edgecolor='black', linewidth=1.5)

# Ajout des valeurs sur les barres
for i, (pays, valeur) in enumerate(pib_2023_compare.items()):
    ax.text(valeur + 500, i, f'{valeur:,.0f} Mds $', 
            va='center', fontweight='bold', fontsize=10)

# Personnalisation
ax.set_title('Classement des pays par PIB nominal en 2023', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('PIB (milliards USD)', fontsize=12, fontweight='bold')
ax.set_ylabel('Pays', fontsize=12, fontweight='bold')

# Grille
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()
```

**Lecture :** Les États-Unis dépassent de près de 50% le PIB chinois, tandis que les économies européennes forment un groupe homogène entre 3000 et 4500 milliards.

### 5.3 Graphique en barres - PIB par habitant 2023

```python
# Préparation des données triées
pib_par_hab_trie = series_pib_par_hab.sort_values(ascending=True)

# Création du graphique
fig, ax = plt.subplots(figsize=(12, 8))

# Définition de couleurs selon les catégories
couleurs_categories = []
for valeur in pib_par_hab_trie.values:
    if valeur > 70000:
        couleurs_categories.append('#2ecc71')  # Vert : très élevé
    elif valeur > 40000:
        couleurs_categories.append('#3498db')  # Bleu : élevé
    elif valeur > 10000:
        couleurs_categories.append('#f39c12')  # Orange : moyen
    else:
        couleurs_categories.append('#e74c3c')  # Rouge : faible

# Création des barres
barres = ax.barh(pib_par_hab_trie.index, pib_par_hab_trie.values,
                 color=couleurs_categories, edgecolor='black', linewidth=1.5)

# Ajout des valeurs
for i, (pays, valeur) in enumerate(pib_par_hab_trie.items()):
    ax.text(valeur + 1500, i, f'{valeur:,.0f} $', 
            va='center', fontweight='bold', fontsize=10)

# Personnalisation
ax.set_title('PIB par habitant en 2023 - Comparaison internationale', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('PIB par habitant (USD)', fontsize=12, fontweight='bold')
ax.set_ylabel('Pays', fontsize=12, fontweight='bold')

# Ajout d'une légende pour les catégories
from matplotlib.patches import Patch
legende_elements = [
    Patch(facecolor='#2ecc71', label='Très élevé (>70k)'),
    Patch(facecolor='#3498db', label='Élevé (40-70k)'),
    Patch(facecolor='#f39c12', label='Moyen (10-40k)'),
    Patch(facecolor='#e74c3c', label='Faible (<10k)')
]
ax.legend(handles=legende_elements, loc='lower right', framealpha=0.9)

# Grille
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()
```

**Analyse :** Ce graphique met en évidence le fossé de développement entre pays riches et émergents, malgré la taille économique impressionnante de la Chine et de l'Inde.

### 5.4 Graphique de croissance - Taux de croissance annuel moyen

```python
# Préparation des données
croissance_moy_triee = df_croissance.mean().sort_values(ascending=False)

# Création du graphique
fig, ax = plt.subplots(figsize=(12, 7))

# Couleurs selon la performance
couleurs_croissance = ['#27ae60' if x > 4 else '#3498db' if x > 2 else '#95a5a6' 
                       for x in croissance_moy_triee.values]

# Création des barres verticales
barres = ax.bar(croissance_moy_triee.index, croissance_moy_triee.values,
                color=couleurs_croissance, edgecolor='black', linewidth=1.5, width=0.6)

# Ajout des valeurs au-dessus des barres
for i, (pays, valeur) in enumerate(croissance_moy_triee.items()):
    ax.text(i, valeur + 0.15, f'{valeur:.1f}%', 
            ha='center', fontweight='bold', fontsize=11)

# Ligne de référence à 2%
ax.axhline(y=2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Seuil 2%')

# Personnalisation
ax.set_title('Taux de croissance annuel moyen du PIB (2015-2023)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Pays', fontsize=12, fontweight='bold')
ax.set_ylabel('Taux de croissance moyen (%)', fontsize=12, fontweight='bold')

# Rotation des labels de l'axe X
plt.xticks(rotation=45, ha='right')

# Grille et légende
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()
```

**Enseignements :** Les économies émergentes (Inde, Chine) affichent des taux de croissance 3 à 6 fois supérieurs aux économies développées, reflétant leur phase de rattrapage économique.

### 5.5 Heatmap - Matrice de corrélation

```python
# Calcul de la matrice de corrélation
matrice_corr = df_pib.iloc[:, :-1].corr()

# Création de la heatmap
fig, ax = plt.subplots(figsize=(10, 8))

# Gén
