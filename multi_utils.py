import operator
import numpy as np


HYP_REF = (11, 11) # reference point for hypervolume (do not change - results will not be comparable)

def hypervolume(pop, ref=HYP_REF):
    non_dom = [pop[i] for i in get_first_nondominated(pop)]

    f1_sorted = list(sorted(non_dom, key=lambda x: x.fitness[0]))

    volume = 0

    for (i, j) in zip(f1_sorted, f1_sorted[1:]):
        volume += (ref[1] - i.fitness[1])*(j.fitness[0] - i.fitness[0])
    
    volume += (ref[1] - f1_sorted[-1].fitness[1])*(ref[0] - f1_sorted[-1].fitness[0])

    return volume


def hypervolume_front(front, ref=HYP_REF):
    volume = 0
    front_sorted = list(sorted(front, key=lambda x: x.fitness[0]))
    for (i, j) in zip(front_sorted, front_sorted[1:]):
        volume += (ref[1] - i.fitness[1]) * (j.fitness[0] - i.fitness[0])

    volume += (ref[1] - front_sorted[-1].fitness[1]) * (ref[0] - front_sorted[-1].fitness[0])
    return volume


def hypervolume_ind(ind, ref=HYP_REF):
    # Extract objective values
    f1 = ind.fitness[0]
    f2 = ind.fitness[1]

    # Calculate the hypervolume as the area of the rectangle
    hv = (ref[0] - f1) * (ref[1] - f2)
    return hv


def assign_crowding_distances(front):
    front = list(sorted(front, key=operator.attrgetter('fitness')))
    front[0].ssc = np.inf # first and last one have infinite crowding distance
    front[-1].ssc = np.inf
    for i in range(1, len(front) - 1):
        front[i].ssc = (front[i + 1].fitness[0] - front[i - 1].fitness[0] +
                        front[i - 1].fitness[1] - front[i + 1].fitness[1])


def assign_hv_contributions(front, ref_point=HYP_REF):
    # Step 0: Handle the edge case where there's only one individual in the front
    if len(front) == 1:
        front[0].ssc = hypervolume_ind(front[0], ref_point)
        return front

    # Step 1: Initialize a list to store hypervolume contributions (ssc)
    total_hv = hypervolume_front(front)

    # Step 2: Calculate the hypervolume contribution for each individual
    for ind in front:
        # Create a new front excluding the current individual
        front_without_ind = [other for other in front if other != ind]
        hv_without_ind = hypervolume_front(front_without_ind, ref_point)

        # Calculate hypervolume contribution as the difference
        ind.ssc = total_hv - hv_without_ind

    return front


def assign_hv_contributions_elite(front, ref_point=HYP_REF):
    # Step 0: Handle the edge case where there's only one individual in the front
    if len(front) == 1:
        front[0].ssc = hypervolume_ind(front[0], ref_point)
        return front

    # Step 1: Sort the front based on the first objective
    front_sorted = sorted(front, key=lambda x: x.fitness[0])

    # Step 2: Assign infinite contribution to the boundary individuals
    front_sorted[0].ssc = float('inf')  # First individual (boundary)
    front_sorted[-1].ssc = float('inf')  # Last individual (boundary)

    # Step 3: Calculate the hypervolume contribution for the rest of the individuals
    total_hv = hypervolume_front(front_sorted, ref_point)

    for i in range(1, len(front_sorted) - 1):  # Skip the boundary individuals
        # Create a new front excluding the current individual
        front_without_ind = front_sorted[:i] + front_sorted[i + 1:]
        hv_without_ind = hypervolume_front(front_without_ind, ref_point)

        # Calculate the hypervolume contribution as the difference
        front_sorted[i].ssc = total_hv - hv_without_ind

    return front_sorted


# returns true if i1 dominates i2
def dominates(fits1, fits2):
    return (all(map(lambda x: x[0] <= x[1], zip(fits1, fits2))) and 
            any(map(lambda x: x[0] < x[1], zip(fits1, fits2))))

def get_first_nondominated(pop):
    non_dom = []
    for i, p in enumerate(pop):
        if not any(map(lambda x: dominates(x.fitness, pop[i].fitness), pop)):
            non_dom.append(i)
    return non_dom

def divide_fronts(pop):
    fronts = []
    while pop:
        non_dom = get_first_nondominated(pop)
        front = [pop[i] for i in non_dom]        
        pop = [p for i,p in enumerate(pop) if i not in non_dom]
        fronts.append(front)
    return fronts