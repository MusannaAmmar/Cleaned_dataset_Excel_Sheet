import numpy as np
import random
import pandas as pd
import os
from config.settings import DATA_DIR
from subsets.brute_force import prepare_data

class SubsetSumGA:
    def __init__(self, numbers, targets, pop_size=100, gen_count=500, mut_rate=0.05, tourn_size=3, elite_size=5):
        """
        Initialize the Genetic Algorithm for subset sum problem.
        
        Parameters:
        - numbers: List or array of numbers to select subsets from.
        - targets: List of target sums to achieve.
        - pop_size: Population size for GA.
        - gen_count: Number of generations.
        - mut_rate: Mutation probability per gene.
        - tourn_size: Number of individuals in tournament selection.
        - elite_size: Number of best individuals to keep each generation.
        """
        self.numbers = np.array(numbers)
        self.targets = targets
        self.pop_size = pop_size
        self.gen_count = gen_count
        self.mut_rate = mut_rate
        self.tourn_size = tourn_size
        self.elite_size = elite_size
        
        # Pre-calculate some statistics for better initialization
        self.total_sum = np.sum(self.numbers)
        self.avg_number = np.mean(self.numbers)

    def _fuzzy_ratio(self, achieved_sum, target):
        """
        Compute a numeric fuzzy ratio between target and achieved sum.
        Returns a value between 0 and 100.
        """
        diff = abs(target - achieved_sum)
        return max(0, 100 * (1 - diff / max(abs(target), 1)))
    

    def _create_individual(self, target=None):
        """Generate a smarter initial chromosome based on target."""
        if target is None:
            # Random initialization
            return np.random.randint(0, 2, len(self.numbers))
        
        # Smart initialization: bias towards numbers that could help reach target
        chromosome = np.zeros(len(self.numbers), dtype=int)
        
        # Estimate how many numbers we might need
        if self.avg_number > 0 and target > 0:
            estimated_count = max(1, int(abs(target) / self.avg_number))
            
            # Add some randomness but ensure positive value
            random_adjustment = random.randint(-2, 3)
            final_count = max(1, min(estimated_count + random_adjustment, len(self.numbers)))
            
            # Ensure we don't exceed array bounds and have positive count
            final_count = max(1, min(final_count, len(self.numbers)))
            
            # Randomly select approximately that many numbers
            indices = np.random.choice(len(self.numbers), 
                                     final_count, 
                                     replace=False)
            chromosome[indices] = 1
        else:
            # Fallback: random initialization if calculation fails
            return np.random.randint(0, 2, len(self.numbers))
        
        return chromosome

    # def _fitness(self, chromosome, target):
    #     """
    #     Improved fitness function with penalty for very large/small subsets.
    #     """
    #     subset_sum = np.dot(chromosome, self.numbers)
        
    #     # Primary fitness: negative absolute difference
    #     diff = abs(subset_sum - target)
    #     fitness = -diff
        
    #     # Bonus for exact match
    #     if diff == 0:
    #         fitness += 1000
        
    #     # Small penalty for very large subsets (encourages smaller solutions)
    #     subset_size = np.sum(chromosome)
    #     if subset_size > len(self.numbers) * 0.7:  # If using more than 70% of numbers
    #         fitness -= 10
    #     elif subset_size == 0:  # Penalty for empty subsets
    #         fitness -= 100
            
    #     return fitness
    def _fitness(self, chromosome, target):
        """
        Fitness now considers fuzzy ratio directly.
        Higher fuzzy ratio => better fitness.
        """
        subset_sum = np.dot(chromosome, self.numbers)
        fuzzy = self._fuzzy_ratio(subset_sum, target)

        # Primary fitness: proportional to fuzzy ratio
        fitness = fuzzy

        # Bonus for exact match
        if subset_sum == target:
            fitness += 1000  # Keep exact match very strong

        # Small penalty for very large subsets (encourages smaller solutions)
        subset_size = np.sum(chromosome)
        if subset_size > len(self.numbers) * 0.7:
            fitness -= 10
        elif subset_size == 0:
            fitness -= 100

        return fitness


    def _selection(self, population, fitnesses):
        """Improved tournament selection with pressure towards better solutions."""
        # Increase tournament size for later generations to increase selection pressure
        tournament_size = min(self.tourn_size, len(population))
        
        idx = random.sample(range(len(population)), tournament_size)
        best_idx = max(idx, key=lambda i: fitnesses[i])
        return population[best_idx].copy()

    def _crossover(self, parent1, parent2):
        """Improved crossover: uniform crossover with some probability."""
        if random.random() < 0.7:  # Single-point crossover
            point = random.randint(1, len(parent1) - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
        else:  # Uniform crossover
            mask = np.random.randint(0, 2, len(parent1))
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            
        return child1.astype(int), child2.astype(int)

    def _mutate(self, chromosome, target=None):
        """
        Improved mutation with adaptive rate and smart bit flipping.
        """
        chromosome = chromosome.copy()
        
        # Adaptive mutation: if we're far from target, mutate more
        if target is not None:
            current_sum = np.dot(chromosome, self.numbers)
            diff_ratio = abs(current_sum - target) / max(target, 1)
            adaptive_rate = min(self.mut_rate * (1 + diff_ratio), 0.3)
        else:
            adaptive_rate = self.mut_rate
        
        # Standard bit flip mutation
        for i in range(len(chromosome)):
            if random.random() < adaptive_rate:
                chromosome[i] = 1 - chromosome[i]
        
        # Smart mutation: occasionally try to add/remove numbers strategically
        if random.random() < 0.1 and target is not None:  # 10% chance of smart mutation
            current_sum = np.dot(chromosome, self.numbers)
            diff = target - current_sum
            
            if diff > 0:  # Need to add more
                # Find unused numbers close to the difference
                unused_indices = np.where(chromosome == 0)[0]
                if len(unused_indices) > 0:
                    # Choose number closest to what we need
                    unused_numbers = self.numbers[unused_indices]
                    best_idx = unused_indices[np.argmin(np.abs(unused_numbers - diff))]
                    chromosome[best_idx] = 1
            elif diff < 0:  # Need to remove some
                # Find used numbers close to the excess
                used_indices = np.where(chromosome == 1)[0]
                if len(used_indices) > 0:
                    used_numbers = self.numbers[used_indices]
                    best_idx = used_indices[np.argmin(np.abs(used_numbers + diff))]
                    chromosome[best_idx] = 0
        
        return chromosome

    def run_ga_for_target(self, target, verbose=False):
        """
        Run GA to find a subset summing to the target.
        """
        # Initialize population with smart initialization
        population = []
        for _ in range(self.pop_size):
            if random.random() < 0.7:  # 70% smart initialization
                individual = self._create_individual(target)
            else:  # 30% random initialization
                individual = self._create_individual()
            population.append(individual)
        
        best_fitness = float('-inf')
        generations_without_improvement = 0
        
        for generation in range(self.gen_count):
            fitnesses = [self._fitness(ind, target) for ind in population]
            
            current_best_fitness = max(fitnesses)
            
            # Check for exact solution
            if current_best_fitness >= 1000:  # Exact match found
                best_idx = np.argmax(fitnesses)
                if verbose:
                    print(f"Exact solution found at generation {generation}")
                return population[best_idx], np.dot(population[best_idx], self.numbers)
            
            # Track improvement
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # Early stopping if no improvement for too long
            if generations_without_improvement > 100:
                if verbose:
                    print(f"Early stopping at generation {generation}")
                break
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate rest of population through crossover and mutation
            while len(new_population) < self.pop_size:
                parent1 = self._selection(population, fitnesses)
                parent2 = self._selection(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                
                child1 = self._mutate(child1, target)
                child2 = self._mutate(child2, target)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            population = new_population[:self.pop_size]
            
            if verbose and generation % 50 == 0:
                best_idx = np.argmax(fitnesses)
                current_sum = np.dot(population[best_idx], self.numbers)
                print(f"Generation {generation}: Best sum = {current_sum}, Target = {target}, Diff = {abs(current_sum - target)}")
        
        # Return best solution found
        fitnesses = [self._fitness(ind, target) for ind in population]
        best_idx = np.argmax(fitnesses)
        return population[best_idx], np.dot(population[best_idx], self.numbers)

    def find_subsets(self, verbose=False):
        """
        Find subsets for all targets.
        """
        results = []
        for i, target in enumerate(self.targets):
            if verbose:
                print(f"\nSolving for target {i+1}/{len(self.targets)}: {target}")
            
            best_chrom, achieved_sum = self.run_ga_for_target(target, verbose)
            subset = self.numbers[best_chrom == 1].tolist()
            exact_match = (achieved_sum == target)
            results.append((target, subset, achieved_sum, exact_match))
            
            if verbose:
                print(f"Result: Sum = {achieved_sum}, Exact = {exact_match}")
        
        return results

    def print_results(self):
        """Print results for all targets."""
        results = self.find_subsets(verbose=True)
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        
        for target, subset, achieved_sum, exact_match in results:
            print(f"\nTarget: {target}")
            # print(f"Subset: {subset}")
            print(f"Achieved Sum: {achieved_sum}")
            print(f"Difference: {abs(achieved_sum - target)}")
            # diff = abs(achieved_sum - target)
            fuzzy_ratio = self._fuzzy_ratio(achieved_sum, target)
            print(f"Fuzzy Ratio: {fuzzy_ratio:.2f}%")
            print(f"Exact Match: {'YES' if exact_match else 'NO'}")
            print(f"Subset Size: {len(subset)}")
            # if len(subset) > 0:
            #     print(f"Subset Numbers: {', '.join(map(str, sorted(subset)))}")


# Main execution
if __name__ == "__main__":
    transactions_file = os.path.join(DATA_DIR, "Customer_Ledger_Entries_FULL.xlsx")
    targets_file = os.path.join(DATA_DIR, "KH_Bank.xlsx")

    # Load and prepare data
    trans_df, target_df = prepare_data(transactions_file, targets_file)
    
    # Use more data for better testing (increase from 1 to 10 rows)
    chunk_trans_data = trans_df.head(3396)  # Increased sample size
    chunk_target_data = target_df.head(1)   # More targets to test
    
    print(f"Transaction data shape: {chunk_trans_data.shape}")
    print(f"Target data shape: {chunk_target_data.shape}")
    
    # Extract numbers and targets
    try:
        amount_cols_trans = [col for col in chunk_trans_data.columns if 'amount' in col.lower()]
        amount_cols_target = [col for col in chunk_target_data.columns if 'amount' in col.lower()]
        
        print(f"Found transaction amount columns: {amount_cols_trans}")
        print(f"Found target amount columns: {amount_cols_target}")

        numbers = pd.to_numeric(
            chunk_trans_data[amount_cols_trans].values.flatten(),
            errors='coerce')

        targets = pd.to_numeric(
            chunk_target_data[amount_cols_target].values.flatten(),
            errors='coerce'
        )
        
        # Remove NaN values and zeros
        numbers = numbers[~np.isnan(numbers)]
        targets = targets[~np.isnan(targets)]
        numbers = numbers[numbers != 0]  # Remove zero amounts
        targets = targets[targets != 0]  # Remove zero targets
        
        print(f"Numbers available: {len(numbers)}")
        print(f"Targets to match: {len(targets)}")
        print(f"Sample numbers: {numbers[:10]}")
        print(f"Sample targets: {targets[:5]}")
        
    except KeyError as e:
        print(f"Column not found: {e}. Please specify correct column names.")
        print(f"Available columns in transactions: {list(chunk_trans_data.columns)}")
        print(f"Available columns in targets: {list(chunk_target_data.columns)}")
        exit(1)
    
    # Ensure numbers and targets are valid
    if len(numbers) == 0 or len(targets) == 0:
        print("Error: Empty data in numbers or targets.")
        exit(1)
    
    # Initialize and run GA with improved parameters
    ga = SubsetSumGA(
        numbers=numbers, 
        targets=targets, 
        pop_size=150,      # Increased population
        gen_count=500,     # More generations
        mut_rate=0.05,     # Slightly higher mutation rate
        tourn_size=5,      # Larger tournament
        elite_size=10      # Keep more elite solutions
    )
    
    ga.print_results()



