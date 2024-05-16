import pandas as pd
import numpy as np
import re
import os

def join_lines(lines):
    """
    Joins lines of a play that are split across multiple lines with newline characters
    
    """
    joined_lines = []
    current_line = ''
    for line in lines:
        if line == '\n':
            joined_lines.append(current_line)
            current_line = ''
            continue
        current_line += line[:-1] + ' '

    return joined_lines


def remove_stage_directions(lines):
    """
    Removes stage directions from a line of text
    """
    return [re.sub(r'\s\[.*?\]\s', '', line) for line in lines]
    

def extract_lines_from_play(filepath):
    """
    Extracts the lines from a play for each character and returns them in a pandas array
    """
    # Read
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Clean up text
    lines = join_lines(lines)
    lines = remove_stage_directions(lines)


    character_lines = {}
    line_number = 0

    for line in lines:
        if line.startswith('ACT') or line.startswith('SCENE') or line.startswith(' ACT') or line.startswith(' SCENE') or line.startswith('THE END'):
            continue
        if line.startswith('['):
            continue
        # Use a regular expression to match the pattern of a character's name at the start of a line
        match = re.match(r'^((?:MS|MR|MRS\.\s)?[A-Z\s\.]+)\.\s+(.*)', line)

        # If a match is found, store the character's name and the rest of the line in the dictionary
        if match:
            character = match.group(1).strip()
            text = match.group(2).strip()

            if character not in character_lines:
                character_lines[character] = []

            character_lines[character].append((text, line_number))
            line_number += 1

    # Convert the dictionary to a pandas DataFrame for easier data manipulation
    df = pd.DataFrame([(k, v, i) for k, vs in character_lines.items() for v, i in vs], columns=['Character', 'Line', 'Line Number'])
    return df

def remove_html_tags(text):
    """
    Removes HTML tags from a string
    """
    return re.sub(r'<.*?>', ' ', text)

if __name__ == '__main__':
    # lines = extract_lines_from_play('data/ideal_husband.txt')
    # lines.to_csv('data/ideal_husband_lines.csv', index=False)
    with open('data/golden_rose.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        text = remove_html_tags(text)
        with open('data/golden_rose.txt', 'w', encoding='utf-8') as f:
            f.write(text)

