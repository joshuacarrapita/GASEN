import numpy as np


def cal_pop_fitness(fitness_func, pop):
    fitness = fitness_func(pop)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next
    # #generation.
    # pop.shape = 8 x 100, fitness.shape = 8 x 1, num_parents = 4
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.min(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = np.inf
    return parents

def crossover(parents, offspring_size):
    # parents.shape = (4 x 100)
    offspring = np.empty(offspring_size) # (8 - 4 = 4 x 100)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    crossover_point = int(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1) % parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover):
    # offspring_crossover.shape = (8 - 4 = 4 x 100)
    # Mutation changes genes in each offspring randomly.
    mutation_rate = 0.1
    for i in range(offspring.shape[0]):
        # Iterate over all genes (models)
        for j in range(offspring.shape[1]):
             if np.random.random() < mutation_rate:
                # Add random noise
                random_value = np.random.uniform(-0.1, 0.1)
                offspring[i, j] += random_value
    
    # Clip to keep weights valid [0, 1]
    return np.clip(offspring, 0.0, 1.0) 
