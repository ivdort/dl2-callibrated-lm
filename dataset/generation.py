import random
import csv

def generate_math_dataset(file_name, num_equations=20000):
    # Define the operations
    operations = ['+', '-', '*', '/']
    
    # Generate math equations with answers
    math_dataset = []
    for _ in range(num_equations):
        num1 = random.randint(1, 100)
        num2 = random.randint(1, 100)
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
        
        equation = f"{num1} {operation} {num2} = "
        math_dataset.append((equation, answer))
    
    # Save the dataset to a CSV file
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Equation', 'Answer'])  # Write the header
        writer.writerows(math_dataset)

