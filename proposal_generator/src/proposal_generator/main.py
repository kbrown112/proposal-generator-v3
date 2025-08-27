#!/usr/bin/env python
import sys
import warnings
import os
from datetime import datetime
import fitz

from proposal_generator.crew import ProposalGenerator

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def read_existing_proposals(filename):
        text = ""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "gold_standard_proposals", filename)
        print(file_path)
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()

        return text

def run():
    """
    Run the crew.
    """
    global proposal_title
    global industry
    proposal_title = input("Proposal Title: ")
    industry = input("Industry: ")
    file_content = read_existing_proposals("First_Tee.pdf")
    print(file_content)

    # print(file_content)

    inputs = {
        'current_year': str(datetime.now().year),
        'proposal_title': proposal_title,
        'industry': industry,
        'file_content': file_content
    }
    
    try:
        ProposalGenerator().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'current_year': str(datetime.now().year),
        'proposal_title': proposal_title,
        'industry': industry
    }
    try:
        ProposalGenerator().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        ProposalGenerator().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        'current_year': str(datetime.now().year),
        'proposal_title': proposal_title,
        'industry': industry
    }
    
    try:
        ProposalGenerator().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
