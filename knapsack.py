import random
import matplotlib.pyplot as plt
import numpy as np
###############Parametry#################

max_weight = 90
population_size = 60
crossover_probability = 0.81
mutation_probability = 0.06
generations = 80

#########################################

# przedmioty w formacie [waga:wartosc], 
items = [[3, 12], [8, 6], [16, 12], [14, 7], [7, 17], [2, 12], [2,9], 
		[1,5],[2,5], [3,5], [4,5], [5,5], [16,1], [16,5],[4,2], [3,1]]

print(f"Przedmioty [M:W]{items}")
print(f"\nParametry:")
print(f"Rozmiar populacji: {population_size}")
print(f"Ilość epok: {generations}")
print(f"Maksymalna waga: {max_weight}")
print(f"Prawdopodobieństwo krzyżowania: {crossover_probability}")
print(f"Prawdopodobieństwo mutacji:{mutation_probability}")

#funkcja generjąca populacje
def generate_population(size):
	population = []
	for _ in range(size):
		genes = [0, 1]
		chromosome = []
		for _ in range(len(items)):
			chromosome.append(random.choice(genes))
		population.append(chromosome)
	return population

#funkcja licząca przystosowanie danego osobnika
def calculate_fitness(chromosome):
	total_weight = 0
	total_value = 0
	for i in range(len(chromosome)):
		if chromosome[i] == 1:
			total_weight += items[i][0]
			total_value += items[i][1]
	if total_weight > max_weight:
		return 0
	else:
		return total_value
	
#funkcja licząca wagę danego osobnika
def calculate_weight(chromosome):
	total_weight = 0
	for i in range(len(chromosome)):
		if chromosome[i] == 1:
			total_weight += items[i][0]
	if total_weight > max_weight:
		return 0
	else:
		return total_weight
	
#Funkcja licząca wartość danej populacji
def calculate_population_value(population):
	population_values = []
	for _ in population:
		population_values.append(calculate_fitness(_))
	return population_values

#funkcja licząca przystosowanie danej populacji
def calculate_population_fitness(population):
	fitness_values = []
	for chromosome in population:
		fitness_values.append(calculate_weight(chromosome))
		while True:
			try:
				fitness_values = [float(i)/sum(fitness_values) for i in fitness_values]
				break
			except ZeroDivisionError:
				fitness_values.append(0)
				break
	return fitness_values

#funkcja kalkulująca wagi danej populacji
def calculate_population_weights(population):
	weight_values = []
	for chromosome in population:
		weight_values.append(calculate_weight(chromosome))
	return weight_values


#funkcja wyboru chromosomów w oparciu o najlepszy wynik wagi
def select_chromosomes(population):
    pairs = []
    weights = calculate_population_fitness(population)
    num_pairs = 10
    
    if not weights:
        print("Error: Empty weights list")
        return pairs
    try:
        for _ in range(num_pairs):
            rand_parents = random.choices(population, weights=weights, k=2)
            pairs.append(rand_parents)
            
            population = [chromosome for chromosome in population if chromosome not in rand_parents]
            weights = [weight for weight, chromosome in zip(weights, population) if chromosome not in rand_parents]
    except Exception as e:
        print(f"Error: {e}")
    return pairs

# funkcja krzyżowania danej pary
def crossover(parent1, parent2):
	child1 = []
	child2 = []
	crossover_point = random.randint(0, len(items)-1)
	child1 = parent1[0:crossover_point] + parent2[crossover_point:]
	child2 = parent2[0:crossover_point] + parent1[crossover_point:]
	print("Krzyżowanko")
	return child1, child2

#Funkcja do przeprowadzania mutacji danego chromosomu osobnika
def mutate(chromosome):
	mutation_point = random.randint(0, len(items)-1)
	if chromosome[mutation_point] == 0:
		chromosome[mutation_point] = 1
	else:
		chromosome[mutation_point] = 0
	print("Mutowanko")
	return chromosome

# funkcja określająca najlepszego osobnika 
def get_best(population):
	fitness_values = []
	for chromosome in population:
		fitness_values.append(calculate_fitness(chromosome))
	max_value = max(fitness_values)
	max_index = fitness_values.index(max_value)
	return population[max_index]

# funkcja określająca najegorszego osobnika
def get_worst(population):
	fitness_values = []
	for chromosome in population:
		fitness_values.append(calculate_fitness(chromosome))
	min_value = min(fitness_values)
	min_index = fitness_values.index(min_value)
	return population[min_index]


population = generate_population(population_size)
generation_scores_worst = []
generation_scores_best = []
generation_scores_avg = []

for _ in range(generations):
	print(f"Liczba osobników w epoce {len(population)}")
	parent_list = select_chromosomes(population)
	new_population = []
	children_list = []
	individual_population_score = []
	individual_population_weight = []
	for parent in parent_list:
		if(random.uniform(0, 1) < crossover_probability):
			child1, child2 = crossover(parent[0], parent[1])
			if random.uniform(0, 1) < mutation_probability:
				child1 = mutate(child1)
			if random.uniform(0, 1) < mutation_probability:
				child2 = mutate(child2)
			children_list.append(child1)
			children_list.append(child2)
		else:
			child1, child2 = parent[0], parent[1]
			if random.uniform(0, 1) < mutation_probability:
				child1 = mutate(child1)
			if random.uniform(0, 1) < mutation_probability:
				child2 = mutate(child2)
			children_list.append(child1)
			children_list.append(child2)
		new_population = children_list + population[len(children_list):]
	generation_scores_best.append(max(calculate_population_value(population)))
	generation_scores_worst.append(min(calculate_population_value(population)))
	generation_scores_avg.append((int(max(calculate_population_value(population)))+int(min(calculate_population_value(population))))/2)
	population = new_population[:]

# określenie najlepszego indywidualnego wyniku w danej grupie osobników
individual_population_score = calculate_population_fitness(population)
individual_population_weight = calculate_population_weights(population)
#wyznaczenie najlepszego/najgorszego osobnika 
best = get_best(population)
worst = get_worst(population)
#wypisanie wyników, najlepszy, najgorszy oraz średnia
print(f"Najlepszy wynik w epoce: {generation_scores_best}")
print(f"Średni wynik w epoce: {generation_scores_avg}")
print(f"Najgorszy wynik w epoce: {generation_scores_worst}")


x = np.arange(start=1, stop=len(generation_scores_avg), step=1)
line_chart1 = plt.plot(generation_scores_worst, color='Blue')
line_chart2 = plt.plot(generation_scores_avg, color='Red')
line_chart3 = plt.plot(generation_scores_best, color='Gray')
plt.xlabel('Ilość generacji',color="black")
plt.ylabel('Wartość plecaka',color="black")
plt.title('Wykres numer generacji/wartość plecaka')
plt.show()