import os
import pandas as pd
import openai
import json

# Load printer data from an Excel file
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data