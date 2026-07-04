from math import sqrt,pow

def euclidean_distance(p1,p2):
    return sqrt(
        pow((p1.x - p2.x),2) + pow((p1.y - p2.y),2)
    )