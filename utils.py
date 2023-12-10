import uuid
import datetime
import re

def get_gene_id(prefix="GEN", include_timestamp=True):
    """
    Generates a custom genetic ID.

    :param prefix: A string prefix for the ID.
    :param include_timestamp: Boolean to decide if a timestamp should be included.
    :return: A string representing the custom genetic ID.
    """
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S") if include_timestamp else ""

    # Construct the genetic ID
    genetic_id = f"{prefix}_{timestamp}"

    return genetic_id

def get_acc(output):
    # Regular expression to find the accuracy
    # This will match a pattern like 'Accuracy of the model on test images: 48.34%'
    match = re.search(r'Accuracy of the model on test images: (\d+\.\d+)%', output)

    if match:
        # Extract the accuracy and convert it to a float
        accuracy = float(match.group(1))
        return accuracy
    else:
        print('Accuracy not found in the output.')
        return None
    
def get_code(response):
    # Define the start and end markers for the code segment
    start_marker = "\\begin{code}"
    end_marker = "\\end{code}"

    # Find the start and end indices of the code segment
    start_index = response.find(start_marker) + len(start_marker)
    end_index = response.find(end_marker, start_index)

    # Extract and return the code segment
    if start_index > -1 and end_index > -1:
        return response[start_index:end_index].strip()
    else:
        return None