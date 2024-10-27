import os
import csv
import sys
import chardet


def detect_encoding(file_path):
    """
    Detect the encoding of a given file.
    
    Args:
    - file_path (str): Path to the file.
    
    Returns:
    - str: Detected encoding of the file.
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']

def extract_values(file_path):
    """
    Extract values following specific strings in a text file.
    
    Args:
    - file_path (str): Path to the text file.
    
    Returns:
    - dict: A dictionary with the extracted values.
    """
    results = {
        "Message to TX": "",
        "System Transactions Throughput": "",
        "Retransmission Success Rate": ""
    }
    
    encoding = detect_encoding(file_path)
    
    with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("# Message to TX:"):
                results["Message to TX"] = round(int(line.split(":")[1].strip()) / 300, 3)
            elif line.startswith("System Transactions Throughput:"):
                results["System Transactions Throughput"] = line.split(":")[1].strip()
            elif line.startswith("Retransmission Success Rate:"):
                results["Retransmission Success Rate"] = line.split(":")[1].strip()
    
    return results

def process_folder(folder_path):
    """
    Process all text files in a given folder and save the results to a CSV file.
    
    Args:
    - folder_path (str): Path to the folder containing text files.
    """
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            extracted_values = extract_values(file_path)
            extracted_values["filename"] = filename
            try :
                method, _, _, _, veh_number, seed = filename.split("_")
                extracted_values["method"] = method
                extracted_values["veh_number"] = veh_number.split("veh")[0]
                extracted_values["seed"] = seed.split(".txt")[0]
            except:
                extracted_values["method"] = ""
                extracted_values["veh_number"] = ""
                extracted_values["seed"] = ""
            data.append(extracted_values)
    
    if data:
        output_file = os.path.join(folder_path, "output.csv")
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ["method", "veh_number", "seed", "Message to TX", "System Transactions Throughput", "Retransmission Success Rate", "filename"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        
        print(f"Data has been written to {output_file}")
    else:
        print("No .txt files found in the specified folder.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_txt_info.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print("The specified path is not a folder.")
        sys.exit(1)
    
    process_folder(folder_path)
