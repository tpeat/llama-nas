import os
from model_base import LlamaModel
from utils import get_gene_id, get_acc, get_code
import subprocess
from deap import base, creator, tools
from datetime import datetime
import random

model = LlamaModel.get_instance()


def run_query(prompt, output_path):
    output = model(prompt=prompt, max_tokens=2048, temperature=0.2)
    print(output)
    # get code from top choice
    text = output['choices'][0]['text']
    code = get_code(text)

    if code is None:
        print("Failed\n", output)
        return

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the output to the specified path
    with open(output_path, 'w') as file:
        file.write(code)


def crossover(gene_id1, gene_id2):
    model1_path = os.path.join('models', gene_id1 + '.py')
    model2_path = os.path.join('models', gene_id2 + '.py')

    with open(model1_path, 'r') as f:
        model1 = f.read()

    with open(model2_path, 'r') as f:
        model2 = f.read()

    query_path = 'queries/mate.txt'
    with open(query_path, 'r') as f:
        query_text = f.read()

    prompt = query_text.format(model1, model2)
    print(prompt)

    new_id = get_gene_id()

    output_path = os.path.join('models', new_id + '.py')

    run_query(prompt, output_path)

# crossover('model_0', 'model_1')
    

def mutate(gene_id):
    model1_path = os.path.join('models', gene_id + '.py')

    with open(model1_path, 'r') as f:
        model1 = f.read()

    query_path = 'queries/mutate.txt'
    with open(query_path, 'r') as f:
        query_text = f.read()

    prompt = query_text.format(model1)
    print(prompt)

    new_id = get_gene_id()

    # overwrite model? 
    output_path = os.path.join('models', new_id)

    run_query(prompt, output_path)

# mutate('model_0')

def eval_ind(gene_id):

    query_path = 'queries/train.txt'
    with open(query_path, 'r') as f:
        query_text = f.read()

    train = query_text.replace('{gene_id}', gene_id)

    run_path = f'runs/{gene_id}.py'

    with open(run_path, 'w') as f:
        f.write(train)
    
    result = subprocess.run(['python', run_path], capture_output=True, text=True)

    # parse accuracy from standard out
    accuracy = get_acc(result.stdout)
    if accuracy is None:
        print("Failed")
        # Define the error log directory and file
        err_log_dir = 'err_log'
        os.makedirs(err_log_dir, exist_ok=True)  # Create directory if it doesn't exist
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(err_log_dir, f'error_log_{timestamp}.txt')

        # Save the output to the error log file
        with open(log_file_path, 'w') as file:
            file.write("Standard Output:\n")
            file.write(result.stdout)
            file.write("\nStandard Error:\n")
            file.write(result.stderr)

    print("Accuracy for", gene_id, "is", accuracy)

    return accuracy,
    

# eval_ind('GEN_20231210174213')

def create_individual(ind_class):
    """
    Create a new individual represented by a unique gene ID.

    :param ind_class: The class used by DEAP to create an individual.
    :return: An instance of ind_class representing the new individual.
    """
    # Generate a new unique gene ID
    gene_id = get_gene_id()

    model_path = os.path.join('models', gene_id + '.py')

    query_path = 'queries/create.txt'

    with open(query_path, 'r') as f:
        prompt = f.read()

    run_query(prompt, model_path)

    # Create a new individual with this gene ID
    individual = ind_class(file_id=gene_id)

    return individual


# Step 4: Setup DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax, file_id=None)

toolbox = base.Toolbox()

toolbox.register("individual", create_individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# custom methods
toolbox.register("evaluate", eval_ind)
toolbox.register("mate", crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selNSGA2)

# Step 5: Run the Genetic Algorithm Loop
def main():
    pop = toolbox.population(n=50)
    CXPB, MUTPB, NGEN = 1, 0.5, 40

    for gen in range(NGEN):

        print("Generation:", gen)
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring

    best_ind = tools.selBest(pop, 1)[0]
    return best_ind

if __name__ == "__main__":
    best_model = main()
    print("Best Model:", best_model)