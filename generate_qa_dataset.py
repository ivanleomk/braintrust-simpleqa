import braintrust
import csv

# Read the local CSV file
with open("./simple_qa_test_set.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header row

    # Initialize Braintrust dataset
    dataset = braintrust.init_dataset(project="SimpleQA", name="SimpleQA")

    # Insert each row into the dataset
    for row in reader:
        metadata = eval(row[0])  # Convert string dict to actual dict
        dataset.insert(
            input=row[1],  # Question
            expected=row[2],  # Answer
            metadata=metadata,
        )

print(dataset.summarize())
