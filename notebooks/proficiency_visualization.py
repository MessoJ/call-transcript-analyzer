# This script uses graphviz to generate a visualization of the CEFR proficiency levels, ensure it is installed.
from graphviz import Digraph
import os

def generate_proficiency_graph(output_filename="proficiency_levels"):
    """
    Generates a Graphviz diagram illustrating CEFR levels and saves it.
    """
    dot = Digraph(comment='CEFR Proficiency Levels')

    # To define nodes with descriptions
    dot.node('A1', 'Basic (A1)\n≤500 unique words\nSimple sentences\nFrequent pauses\n≥30% errors')
    dot.node('A2', 'Elementary (A2)\n500-1k words\nBasic compound sentences\nRegular pauses\n20-30% errors')
    dot.node('B1', 'Intermediate (B1)\n1k-2k words\nSome complex sentences\nModerate fluency\n10-20% errors')
    dot.node('B2', 'Upper-Intermediate (B2)\n2k-3.5k words\nRegular complex structures\nGood fluency\n5-10% errors')
    dot.node('C1', 'Advanced (C1)\n3.5k-5k words\nVarious complex structures\nNear-natural fluency\n<5% errors')
    dot.node('C2', 'Proficient (C2)\n>5k words\nSophisticated structures\nNative-like fluency\nMinimal errors')

    # To define edges showing progression
    dot.edges([
        ('A1', 'A2'),
        ('A2', 'B1'),
        ('B1', 'B2'),
        ('B2', 'C1'),
        ('C1', 'C2')
    ])

    # To render the graph to a file (e.g., PNG, PDF)
    try:
        
        dot.render(output_filename, view=False, format='png', cleanup=True)
        print(f"Proficiency graph saved as {output_filename}.png")
        # You may change format='pdf' or other supported formats if you like, whatever!
    except Exception as e:
        print(f"Error rendering graph: {e}")
        print("Please ensure Graphviz is installed and accessible in your system's PATH.")

if __name__ == "__main__":
    if not os.path.exists("notebooks"):
        os.makedirs("notebooks")
    generate_proficiency_graph(output_filename="notebooks/proficiency_levels")
