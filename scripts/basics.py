name = "Developer_Alpha"
age = 24
marks = [85, 92, 78, 95, 88]

for i in marks:
    print(f"Validating Data Point: {i}")

def analyze_numbers(numbers):
    min_val = min(numbers)
    max_val = max(numbers)
    avg_val = sum(numbers) / len(numbers)
    return min_val, max_val, avg_val

results = analyze_numbers(marks)

print("\n--- Statistics Report ---")
print(f"Low: {results[0]} | High: {results[1]} | Average: {results[2]}")