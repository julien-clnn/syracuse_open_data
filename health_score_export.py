import osmnx as ox
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Étape 1 : Charger les données des quartiers de Syracuse
# Téléchargez le fichier shapefile depuis le lien fourni et extrayez-le dans un répertoire local
shapefile_path = 'inputs/Syracuse_Neighborhoods.shp'
quartiers = gpd.read_file(shapefile_path)

# Vérifier le système de coordonnées et le reprojeter si nécessaire
if quartiers.crs != 'EPSG:4326':
    quartiers = quartiers.to_crs(epsg=4326)

# Étape 2 : Collecter les données des installations de santé
# Définir les types d'installations de santé à rechercher
amenities = ['hospital', 'clinic', 'doctors']

# Obtenir les limites de la ville de Syracuse pour restreindre la zone de recherche
ville = ox.geocode_to_gdf('Syracuse, New York, USA')
ville = ville.to_crs(epsg=4326)
limites_ville = ville.geometry.unary_union

# Télécharger les données des installations de santé dans les limites de la ville
tags = {'amenity': amenities}
pois = ox.features_from_polygon(limites_ville, tags)

# Filtrer les points d'intérêt pour ne conserver que les installations de santé
pois_sante = pois[pois['amenity'].isin(amenities)]

# Étape 3 : Calculer un score de santé pour chaque quartier
# Initialiser une colonne pour le score
quartiers['score_sante'] = 0

# Filtrer les points d'intérêt pour ne conserver que les hôpitaux et cliniques
hospi_sante = pois_sante[pois_sante['amenity'].isin(['hospital', 'clinic'])]

# Filtrer les points d'intérêt pour ne conserver que les médecins
doctors_sante = pois_sante[pois_sante['amenity'] == 'doctors']

import numpy as np
from shapely.geometry import box
from scipy.spatial import cKDTree

# Convertir toutes les géométries en points en utilisant le centroid si nécessaire
hospi_sante = hospi_sante.copy()
hospi_sante['geometry'] = hospi_sante.geometry.apply(lambda geom: geom.centroid if geom.geom_type != 'Point' else geom)

# Préparer les coordonnées des hôpitaux et cliniques pour une recherche rapide
hospi_coords = np.array([[point.x, point.y] for point in hospi_sante.geometry])
hospi_tree = cKDTree(hospi_coords)

# Parcourir chaque quartier pour calculer le score de santé
for idx, quartier in quartiers.iterrows():
    # Obtenir les limites du quartier
    minx, miny, maxx, maxy = quartier.geometry.bounds

    # Générer des cellules de 100m x 100m dans le quartier
    x_grid = np.arange(minx, maxx, 0.001)  # Ajuster l'échelle si nécessaire
    y_grid = np.arange(miny, maxy, 0.001)  # Ajuster l'échelle si nécessaire
    grid_cells = [box(x, y, x + 0.001, y + 0.001) for x in x_grid for y in y_grid]
    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=quartiers.crs)

    # Initialiser le score d'influence pour ce quartier
    score_influence = 0

    # Parcourir chaque cellule de la grille
    for cell in grid.geometry:
        # Vérifier si un hôpital ou une clinique est dans la cellule
        hop_in_cell = hospi_sante[hospi_sante.within(cell)]
        if not hop_in_cell.empty:
            score_influence += 1
        else:
            # Calculer la distance au hôpital ou à la clinique le plus proche
            centroid = cell.centroid
            distance, _ = hospi_tree.query([centroid.x, centroid.y])
            if distance > 0:
                score_influence += 1 / distance

    # Compter le nombre de médecins dans le quartier
    number_of_doctors = doctors_sante[doctors_sante.within(quartier.geometry)].shape[0]

    # Multiplier le score d'influence par le nombre de médecins
    score_influence *= np.log(max(number_of_doctors, 3))

    # Mettre à jour le score d'influence du quartier
    quartiers.at[idx, 'score_influence'] = score_influence

# Normaliser les scores de santé entre 0 et 1
quartiers['score'] = (quartiers['score_influence'] - quartiers['score_influence'].min()) / (quartiers['score_influence'].max() - quartiers['score_influence'].min())

# Exporter quartiers en shapefile
quartiers.to_file('output/neighboors_health.shp', driver='ESRI Shapefile')