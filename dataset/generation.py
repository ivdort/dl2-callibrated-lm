import random
import csv

def generate_math_dataset(file_name, num_equations=20000, parity=False):
    # Define the operations
    operations = ['+', 
                  '-', 
                  '*', 
                #   '/'
                  ]
    
    # Generate math equations with answers
    math_dataset = []
    for _ in range(num_equations):
        num1 = random.randint(1, 10)
        num2 = random.randint(1, 10)
        operation = random.choice(operations)
        
        if operation == '+':
            answer = num1 + num2
            

        elif operation == '-':
            answer = num1 - num2
        elif operation == '*':
            answer = num1 * num2
        elif operation == '/':
            # Adjust num1 to ensure the division has no remainder
            num1 = num1 * num2
            answer = num1 // num2
        
        # make answer 'even' or 'odd'
        if parity:        
            if answer % 2 == 0:
                answer = 'even'
            else:
                answer = 'odd'

        equation = f"{num1} {operation} {num2} = "
        math_dataset.append((equation, answer))
    
    # Save the dataset to a CSV file
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Equation', 'Answer'])  # Write the header
        writer.writerows(math_dataset)

generate_math_dataset('math_dataset_1_to_10_+_-_*_True.csv', num_equations=20000, parity=True)