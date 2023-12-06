# Projet de MOGPL M1 Informatique 2023-2024

## Binome :

### M1 DAC Yuan Liu 21214576

### M1 DAC Herve Nguyen 28625990


from random import *
from dictionnaireadjacenceorientepondere import DictionnaireAdjacenceOrientePondere
from matplotlib import pyplot as plt
from graphviz import Digraph
from IPython.display import Image, display

def afficher_graphe(graph):
    """
    Affiche un graphe orienté pondéré dans un notebook Jupyter en utilisant Graphviz.
    """
    dot = Digraph(comment='Graphe Orienté Pondéré', format='png')

    # Ajustements pour améliorer la lisibilité
    dot.attr(size='10,10')
    dot.attr(rankdir='LR')  # Pour un affichage de gauche à droite
    dot.attr(dpi='300')     # Résolution plus élevée pour une meilleure qualité

    for sommet in graph.sommets():
        dot.node(str(sommet), shape='circle', fontsize='10')

    for u, v, poids in graph.arcs():
        dot.edge(str(u), str(v), label=str(poids), fontsize='8')

    dot.render('graphe', cleanup=True)
    display(Image(filename='graphe.png'))


# Question 1

def bellmanford(G, source):
    """Implémente l'algorithme de Bellman-Ford pour un graphe orienté pondéré."""
    distance = dict()
    parent = dict()
    iterations = None
    
    for sommet in G.sommets():
        distance[sommet] = float('inf')

    for sommet in G.sommets():
        parent[sommet] = None

    for v in G.successeurs(source):
        distance[v] = G.poids_arc(source, v)
        parent[v] = source


    distance[source] = 0
    for i in range(len(G.sommets())):
        old = distance.copy()
        for (u, v, p) in G.arcs():
            if distance[v] > distance[u] + p:
                if (v == source):
                    afficher_graphe(G)
                    raise Exception
                parent[v] = u
                distance[v] = distance[u] + p
        if old == distance:
            iterations = i + 1
            break
    return distance, parent, iterations

def obtenirChemin(source, destination, parent):
    """Retourne le chemin le plus court entre la source et la destination à partir du tableau des parents."""
    if parent[destination] is None:
        return []
    current = destination
    res = [current]
    while current != source:
        res.append(parent[current])
        current = parent[current]
    res.reverse()
    return res


# Question 2
def sources(G):
    """Retourne une liste des sources du graphe orienté G."""
    res = []
    for sommet in G.sommets():
        if G.degre_entrant(sommet) == 0:
            res.append(sommet)
    return res

def puits(G):
    """Retourne une liste des puits du graphe orienté G."""
    res = []
    for sommet in G.sommets():
        if G.degre_sortant(sommet) == 0:
            res.append(sommet)
    return res

def GloutonFas(G):
    """Implémente l'algorithme GloutonFas pour obtenir un ordre optimal des sommets d'un graphe orienté."""
    s1, s2 = [], []
    while(G.nombre_sommets() > 0):
        liste_source = sources(G)
        while(liste_source):
            u = liste_source[0]
            s1.append(u)
            G.retirer_sommet(u)
            liste_source = sources(G)
        
        liste_puits = puits(G)
        while(liste_puits):
            u = liste_puits[0]
            s2.insert(0, u)
            G.retirer_sommet(u)
            liste_puits = puits(G)
        
        maximum = -float('inf')
        u = None
        for sommet in G.sommets():
            if G.degre_sortant(sommet) - G.degre_entrant(sommet) > maximum:
                maximum = G.degre_sortant(sommet) - G.degre_entrant(sommet)
                u = sommet
        if (u != None):      
            s1.append(u)
            G.retirer_sommet(u)
    
    return s1 + s2



# Question 3
def getGraphes():
    """Retourne quatre graphes orientés pondérés pré-définis."""
    G1 = DictionnaireAdjacenceOrientePondere()
    G2 = DictionnaireAdjacenceOrientePondere()
    G3 = DictionnaireAdjacenceOrientePondere()
    H  = DictionnaireAdjacenceOrientePondere()

    G1.ajouter_sommets([1,2,3,4,5,6,7,8,9,10,11,12])
    G2.ajouter_sommets([1,2,3,4,5,6,7,8,9,10,11,12])
    G3.ajouter_sommets([1,2,3,4,5,6,7,8,9,10,11,12])
    H.ajouter_sommets([1,2,3,4,5,6,7,8,9,10,11,12])

    G1.ajouter_arcs([(1,2,7),(1,3,2),(2,3,8),(3,4,6),(4,5,-2),(4,6,-7),(4,7,1),(5,7,5),(6,5,2),(6,8,5),(7,1,-3),(8,2,10)])
    G1.ajouter_arcs([(8,3,5),(2,10,6),(5,12,-2),(6,9,-5),(6,11,-4),(8,10,3),(8,11,9),(10,9,-2),(10,11,-3),(11,12,6)])


    G2.ajouter_arcs([(1,2,7),(1,3,5),(2,3,-1),(3,4,3),(4,5,-3),(4,6,8),(4,7,-2),(5,7,4),(6,5,2),(6,8,3),(7,1,2),(8,2,-2)])
    G2.ajouter_arcs([(8,3,2),(2,10,2),(5,12,5),(6,9,7),(6,11,10),(8,10,-3),(8,11,-7),(10,9,-1),(10,11,6),(11,12,9)])

    G3.ajouter_arcs([(1,2,-4),(1,3,2),(2,3,8),(3,4,9),(4,5,3),(4,6,1),(4,7,-5),(5,7,-1),(6,5,-6),(6,8,4),(7,1,6),(8,2,7)])
    G3.ajouter_arcs([(8,3,-3),(2,10,-5),(5,12,1),(6,9,-3),(6,11,7),(8,10,2),(8,11,4),(10,9,4),(10,11,-1),(11,12,8)])

    H.ajouter_arcs([(1,2,8),(1,3,-2),(2,3,5),(3,4,3),(4,5,-2),(4,6,-5),(4,7,2),(5,7,1),(6,5,8),(6,8,7),(7,1,5),(8,2,-6)])
    H.ajouter_arcs([(8,3,-4),(2,10,3),(5,12,9),(6,9,5),(6,11,5),(8,10,-7),(8,11,-8),(10,9,6),(10,11,-4),(11,12,1)])

    return G1,G2,G3,H

# Question 4

def obtenirPlusCourtChemins(source, parent, sommets):
    """Retourne une liste de chemins les plus courts à partir d'un tableau de parents."""
    chemins = []
    for destination in sommets:
        if destination != source:
            chemins.append(obtenirChemin(source, destination, parent))
    return chemins


def cheminsToGraphe(T, sommets):
    """Convertit une liste de chemins en un graphe."""
    G = DictionnaireAdjacenceOrientePondere()
    G.ajouter_sommets(sommets)
    for chemin in T:
        for i in range(len(chemin) - 1):
            G.ajouter_arc(chemin[i], chemin[i+1], 0)
    return G

# Question 5



# Question 6

def bellmanford_ordre(G, source, ordre):
    """Implémente l'algorithme de Bellman-Ford avec un ordre spécifique des sommets."""
    distance = dict()
    parent = dict()
    iterations = None
    for sommet in G.sommets():
        distance[sommet] = float('inf')

    for sommet in G.sommets():
        parent[sommet] = None

    for v in G.successeurs(source):
        distance[v] = G.poids_arc(source, v)
        parent[v] = source


    distance[source] = 0

    for i in range(len(G.sommets())):
        old = distance.copy()
        for u in ordre:
            for v in (G.successeurs(u)):
                p = G.poids_arc(u, v)
                if distance[v] > distance[u] + p:
                    parent[v] = u
                    distance[v] = distance[u] + p
        if old == distance:
            iterations = i + 1
            break
    if (iterations == None):
        afficher_graphe(G)
        
    return distance, parent, iterations

# Question 7

def comparaisonOrdre(G, source, ordre, n=10000):
    """Compare les itérations de Bellman-Ford avec un ordre spécifique par rapport à des ordres aléatoires."""
    better = 0
    eq = 0
    worse = 0
    total = 0
    best = float("inf")
    
    _,_,ordre_iter = bellmanford_ordre(G, source, ordre)
    
    for k in range(n):
        random_ordre = list(G.sommets()).copy()
        shuffle(random_ordre)
        _,_,random_iter = bellmanford_ordre(G, source, random_ordre)
        
        if (random_iter > ordre_iter):
            worse += 1
        if (random_iter == ordre_iter):
            eq += 1
        if (random_iter < ordre_iter):
            better += 1
        if (random_iter < best):
            best = random_iter
        total += random_iter
    avg = total / n
    
    print(f"avg = {avg}, best = {best}, gloutonFas = {ordre_iter}, worse = {worse}, eq = {eq}, better = {better}")

# Question 9

def obtenirOrdre(lst):
    """Obtient un ordre optimal des sommets à partir d'une liste de graphes."""
    T = []
    for k in range(len(lst)):
        _, p, _ = bellmanford(lst[k],0)
        T += obtenirPlusCourtChemins(0, p, lst[k].sommets())
    T = cheminsToGraphe(T, lst[0].sommets())
    return GloutonFas(T)

def genererSommets(n):
    """Génère une liste de sommets."""
    return list(range(n))

def select_sommets(sommets):
    """Sélectionne deux sommets distincts de la liste."""
    sommets_tmp = sommets.copy()
    a = choice(sommets_tmp)
    sommets_tmp.remove(a)
    b = choice(sommets_tmp)
    return a, b

def genererArcs(sommets):
    """Génère une liste d'arcs pour un graphe orienté pondéré."""
    nb_arcs = randint(len(sommets), len(sommets) * 2 - 1)
    liste_arcs = [(0,1)]
    for i in range(nb_arcs):
        a, b = select_sommets(sommets)
        while (a,b) in liste_arcs:
            a, b = select_sommets(sommets)
        
        liste_arcs.append((a,b))
    return liste_arcs
        
def genererPoids(liste_arcs):
    """Génère des poids aléatoires pour une liste d'arcs"""
    new = []
    for (a, b) in liste_arcs:
        poids = randint(-10, 10)
        new.append((a,b,poids))
    return new

def genererGraphe(sommets, liste_arcs):
    """Génère un graphe orienté pondéré à partir des paramètres donnés"""
    G = DictionnaireAdjacenceOrientePondere()
    G.ajouter_sommets(sommets)
    G.ajouter_arcs(genererPoids(liste_arcs))
    return G

def genererGrapheSansCycleNeg(sommets, liste_arcs, source=0):
    """Génère un graphe sans cycle négatif à partir des paramètres donnés"""
    G = genererGraphe(sommets, liste_arcs)
    while(G.detecter_cycle_poids_negatifs(source)):
        G = genererGraphe(sommets, liste_arcs)
    return G

def genererEnsembleGraphe(nb_graphe, nb_sommets):
    """Génère un ensemble de graphes sans cycle négatif à partir des paramètres donnés"""
    ensemble = []
    sommets = genererSommets(nb_sommets)
    arcs = genererArcs(sommets)
    for i in range(nb_graphe):
        ensemble.append(genererGrapheSansCycleNeg(sommets, arcs))
    return ensemble

def genererH_ordre(nb_graphe=4, nb_sommets=20):
    """Génère un graphe et un ordre calculé avec GloutonFas à partir des paramètres donnés"""
    instances = genererEnsembleGraphe(nb_graphe, nb_sommets)
    H = instances.pop()
    return H, obtenirOrdre(instances)



# Question 10

def correlation_nb_instance(sommets, arcs, nb_enchantillon=10, nb_instance_max=15, retourner_H=False, verification=True):
    """Analyse la corrélation entre le nombre d'instances fournies à GloutonFas et le nombre d'itérations dans Bellman-Ford sur un modele de graphe donné"""
    x = []
    y = []

    if verification:
        H = genererGrapheSansCycleNeg(sommets, arcs)
    else:
        H = genererGraphe(sommets, arcs)

    for nb_graphes in range(2,nb_instance_max):
        total = 0
        
        for i in range(nb_enchantillon):
            instances = []
            for j in range(nb_graphes):
                if verification:
                    instances.append(genererGrapheSansCycleNeg(sommets, arcs))
                else :
                    instances.append(genererGraphe(sommets, arcs))
            ordre = obtenirOrdre(instances)
            _, _, iteration = bellmanford_ordre(H, 0, ordre)
            total += iteration

        avg = total / nb_enchantillon
        y.append(avg)
        x.append(nb_graphes + 1)
    
    plt.plot(x, y, "r", label="nombre d'iteration" )
    plt.legend()

    if retourner_H :
        return H


# Question 11

def genererInstance():
    """Génère une instance spécifique de la famille d'instance a plusieurs niveau"""
    n = 2500
    sommets = genererSommets(10000)
    G = DictionnaireAdjacenceOrientePondere()
    G.ajouter_sommets(sommets)
    liste_arcs = []
    for k in range(0, 2500):
        if (k < 2499):
            for i in range(4):
                for j in range(4):
                    liste_arcs.append((k * 4 + i, (k + 1) * 4 + j))
    G.ajouter_arcs(genererPoids(liste_arcs))
    return G

